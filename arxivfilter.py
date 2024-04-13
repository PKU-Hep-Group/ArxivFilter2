import os
import re
import yaml
from datetime import datetime 
from pytz import timezone
import tarfile
# import urllib.request
import wget
import openai
import arxiv
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml'), 'r') as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    logging.debug('config.yaml loaded: \n' + str(CONFIG))


class Query(object):
    def __init__(self, result):
        self.date = result.updated
        self.url = result.entry_id
        self.pdf_url = result.pdf_url
        self.title = result.title
        self.authors = ', '.join([author.name for author in result.authors])
        self.abstract = result.summary.replace('\n', ' ')
        self.published = result.published
        self.id = result.entry_id.split('/')[-1]
        self.browse_html_url = "https://browse.arxiv.org/html/%s" % self.id
        self.categories = result.categories
        self.primary_category = result.primary_category
        self.content_str = self.get_content_str()

    @property
    def is_recent(self):
        curr_time = datetime.now(timezone('GMT'))
        delta_time = curr_time - self.date
        assert delta_time.total_seconds() > 0
        return delta_time.days < 7

    def __hash__(self):
        return self.id

    def __str__(self):
        return self.title + '\n' + self.content_str
    
    def get_content_str(self):
        s = ''
        s += self.authors + '\n'
        s += ', '.join(self.categories) + '\n'
        s += self.date.ctime() + ' GMT \n'
        s += self.url + '\n'
        s += self.pdf_url + '\n'
        s += self.browse_html_url + '\n'
        s += '\n' + self.abstract + '\n'
        return s


class ChatGPT3(object):
    def __init__(self):
        self._opts = CONFIG['openai']
        self.model = self._opts['model']

        openai.api_key = self._opts['api_key']

    def get_llm_response(self, prompt):

        if self._opts['dryrun']:
            return 'This is a dryrun. No actual request is sent to OpenAI API.'

        import subprocess
        set_proxy = False
        if '中国' in subprocess.check_output(['curl', 'cip.cc']).decode('utf-8') and self._opts['optional_user_proxy']:
            logging.debug('Use proxy: ' + self._opts['optional_user_proxy'] + ' for OpenAI API.')
            set_proxy = True
        if set_proxy:
            os.environ["HTTP_PROXY"] = self._opts['optional_user_proxy']
            os.environ["HTTPS_PROXY"] = self._opts['optional_user_proxy']

        try:
            response = openai.ChatCompletion.create(
                model=self._opts['model'],
                max_tokens=self._opts['max_tokens'],
                temperature=self._opts['temperature'],
                messages = [{"role": "user", "content": prompt}]
            )
            return_msg = response.choices[0].message.content
        except Exception as e:
            logging.warning('Error in getting response. ' + str(e))
            return_msg = ''
        
        if set_proxy:
            del os.environ["HTTP_PROXY"], os.environ["HTTPS_PROXY"]
        return return_msg


class ArxivAnalyzer(object):
    def __init__(self, obj):
        self._opts = CONFIG['af_options']
        self._obj = obj
        self._id = obj.id
        self._idnum = obj.id.split('v')[0]
        self._base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), self._idnum)
        os.makedirs(self._base_dir, exist_ok=True)
        self._llm = ChatGPT3()

        self._latex_intro = None
        self._latex_concl = None

        self.llm_response = None
        self.llm_model = self._llm.model


    def get_arxiv_source(self):

        if os.path.exists(os.path.join(self._base_dir, 'source.tar.gz')) and self._opts['reuse_arxiv_source']:
            logging.debug('source.tar.gz already exists. Skip downloading.')
            return

        # download source
        wget.download('https://arxiv.org/e-print/' + self._idnum, out=os.path.join(self._base_dir, 'source.tar.gz'))

        # extract source
        src = tarfile.open(os.path.join(self._base_dir, 'source.tar.gz'))
        src.extractall(self._base_dir)

    def parse_latex_sections(self):
        
        # parse all tex files
        text = ''
        for fname in os.listdir(self._base_dir):
            if fname.endswith('.tex'):
                with open(os.path.join(self._base_dir, fname), 'r') as f:
                    text += f.read()

        # removed commented lines and labels
        text = re.sub(r'^\s*%.*?$', '', text, flags=re.MULTILINE) # The re.MULTILINE flag ensures that ^ and $ match the start and end of each line, respectively.
        text = re.sub(r'\s*\\(label|cite)\{.*?\}', '', text, flags=re.MULTILINE) # remove \label{...} and \cite{...}

        # Find all sections and the content after them
        pattern = r'\\section\*?\{(.*?)\}(.*?)(?=\\section\*?\{|$)'
        matches = re.findall(pattern, text, re.DOTALL)

        sections = {section_name: content.strip() for section_name, content in matches}

        # parse introduction and conclusions
        found_intro = False
        found_concl = False
        for sec_name, content in sections.items():
            if not found_intro and 'introduction' in sec_name.lower():
                self._latex_intro = content
                found_intro = True
            if not found_concl and any(s in sec_name.lower() for s in ['conclusion', 'summary']):
                self._latex_concl = content
                found_concl = True


    def clean_up(self):
        # remove directory
        import shutil
        shutil.rmtree(self._base_dir)


    def get_prompt(self, use_latex_intro=True, use_latex_concl=True):
        _prompt = '''
Please read the abstract, introduction, and conclusion (if valid) of a paper, and answer the following questions: (1) Please list key insights and lessons learned from the paper in Chinese. (2) Act as an experienced researcher in the field of high energy physics and machine learning to provide 3-5 suggestions for related topics or future research directions in Chinese based on the paper's abstract. (3) Same as 2, but this time focus more on the introduction and conclusion to try answering the question again.

---
Abstract:

{abstract}

---
Introduction:

{intro}

---
Conclusion:

{concl}
'''.format(
    abstract = self._obj.abstract,
    intro = self._latex_intro if use_latex_intro and self._latex_intro else '(Not valid)',
    concl = self._latex_concl if use_latex_concl and self._latex_concl else '(Not valid)',
)
        logging.debug('Prompt generated: \n\n' + _prompt)
        return _prompt


    def get_llm_response(self):
        _llm_prompt_opts_retry_list = [
            dict(use_latex_intro=True, use_latex_concl=True),
            dict(use_latex_intro=True, use_latex_concl=False),
            dict(use_latex_intro=False, use_latex_concl=False),
        ]
        for _llm_prompt_opts in _llm_prompt_opts_retry_list:
            self.llm_prompt_opts = _llm_prompt_opts
            prompt = self.get_prompt(**self.llm_prompt_opts)
            self.llm_response = self._llm.get_llm_response(prompt)
            if self.llm_response != '':
                ## success
                break

        return self.llm_response


class ArxivFilter(object):
    def __init__(self):
        self._categories = CONFIG['categories']
        self._keywords = CONFIG['keyword_logic'].replace('\n', ' ')
        self._opts = CONFIG['af_options']
        self._mail = CONFIG['mail']

    @property
    def _previous_arxivs_fname(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'previous_arxivs.txt')
        
    def _get_previously_sent_arxivs(self):
        if os.path.exists(self._previous_arxivs_fname):
            with open(self._previous_arxivs_fname, 'r') as f:
                return set(f.read().split('\n'))
        else:
            return set()

    def _save_previously_sent_arxivs(self, new_queries):
        prev_arxivs = list(self._get_previously_sent_arxivs())
        prev_arxivs += [q.id for q in new_queries]
        prev_arxivs = list(set(prev_arxivs))
        with open(self._previous_arxivs_fname, 'w') as f:
            f.write('\n'.join(prev_arxivs))
        
    def _get_queries_from_last_day(self, max_results=100):
        queries = []

        # get all queries in the categories in the last day
        for category in self._categories:
            new_queries = [
                Query(q) for q in arxiv.Search(
                    query=category,
                    sort_by=arxiv.SortCriterion.LastUpdatedDate,
                    max_results=max_results,
                    ).results()
                ]
            logging.debug('category: ' + category + ', found ' + str(len(new_queries)) + ' papers')

            queries += [q for q in new_queries if q.is_recent]

        logging.debug('number of recent papers in all categories found: ' + str(len(queries)))

        # get rid of duplicates
        queries_dict = {q.id: q for q in queries}
        unique_keys = set(queries_dict.keys())
        queries = [queries_dict[k] for k in unique_keys]

        # only keep queries that contain keywords
        keywords_check_script = self._keywords
        keywords_check_script = re.sub('"{([^"]*)}"', '(\'\g<1>\' in str(q))', keywords_check_script) # keep the upper/lower cases
        keywords_check_script = re.sub('"([^"]*)"', '(\'\g<1>\'.lower() in str(q).lower())', keywords_check_script)
        queries = [q for q in queries if eval(keywords_check_script)]
        logging.debug('keywords : ' + self._keywords)
        logging.debug('papers containing keywords: ' + ', '.join([q.id for q in queries]))

        # sort from most recent to least
        queries = sorted(queries, key=lambda q: (datetime.now(timezone('GMT')) - q.date).total_seconds())

        # filter if previously sent
        prev_arxivs = self._get_previously_sent_arxivs()
        queries = [q for q in queries if q.id not in prev_arxivs]
        logging.debug('new papers to mail: ' + ', '.join([q.id for q in queries]))

        
        return queries

    def _get_query_from_id(self, id):
        return [Query(q) for q in arxiv.Search(id_list=[id], max_results=1).results()][0]


    def _send_email_163(self, text, title, sender=None):
        import smtplib
        from email.mime.text import MIMEText
        from email.header import Header
        from email.utils import formataddr

        mail_host = self._mail['host']
        mail_user = self._mail['user']
        mail_pass = self._mail['password']

        if sender is None:
            sender = 'ArxivFilter'
        receiver = self._mail['receiver']

        message = MIMEText(text, 'plain', 'utf-8')
        message['From'] = formataddr([sender, mail_user])
        message['To'] = receiver
        subject = title
        message['Subject'] = subject
        logging.debug('Title: ' + title)
        logging.debug('Text: \n' + text)

        if self._mail['dryrun']:
            logging.info('This is a dryrun. Email not sent.')
            return

        try:
            smtpObj = smtplib.SMTP_SSL(mail_host, 994)
            smtpObj.login(mail_user, mail_pass)  
            smtpObj.sendmail(mail_user, receiver, message.as_string())
            print('Email sent!')
        except smtplib.SMTPException as e:
            print('Error: cannot send email: ', e)

    def run(self):
        # get queries
        if self._opts['test_mode']:
            queries = [self._get_query_from_id(str(self._opts['test_paper_id']))]
        else:
            queries = self._get_queries_from_last_day()

        ## analyze the content by LLM
        for q in queries:
            aa = ArxivAnalyzer(q)
            if self._opts['need_arxiv_source']:
                aa.get_arxiv_source()
                aa.parse_latex_sections()
            aa.clean_up()
            aa.get_llm_response()
            print(aa.llm_response)
            mail_text_template = '''{content_str}
-----------------------------
Analyze by ChatGPT ({model})
(use_abs=True, use_latex_intro={use_latex_intro}, use_latex_concl={use_latex_concl})

{llm_response}
-----------------------------
Sent from ArxivFilter
'''
            mail_text = mail_text_template.format(
                content_str = q.content_str,
                model = aa.llm_model,
                use_latex_intro = aa.llm_prompt_opts["use_latex_intro"],
                use_latex_concl = aa.llm_prompt_opts["use_latex_concl"],
                llm_response = aa.llm_response,
            )
            self._send_email_163(mail_text, title=q.title, sender=q.primary_category)
        
        self._save_previously_sent_arxivs(queries)


def dailyrun():
    # daily routine. Reload all configurations
    af = ArxivFilter()
    af.run()


if __name__ == '__main__':
    dailyrun()
