import ollama
from bs4 import BeautifulSoup
from .types_ import Article
from .logger import Logger
from .utils import dedent




class NewsPageScraper:
    def __init__(self) -> None:
        self.logger = Logger()

    def _get_clean_text(self, raw_text: str) -> str:
        prompt = dedent(
            """
            ## Raw Text
            {text}

            ## Task
            The raw text is scraped from webpage, which contains an article. 
            Clean it, and return the article formatted as Markdown. 
            Return nothing but the article content - avoid any ads, subscription requests or engagement to follow/like/share/comment.
            Do NOT add any prefix text like "Here's the clean text" - just write the text!
            """)
        text = ollama.chat(
            model='llama3.1',
            options={'temperature': 0.3},
            messages=[{'role': 'user', 'content': prompt.format(text=raw_text)}],
        )['message']['content']
        return text
    
    def from_html(self, html: str, url: str) -> Article:
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else None
        raw_text = soup.get_text()
        text = self._get_clean_text(raw_text)
        return Article(
            url=url,
            title=title or '',
            text=text
        )
    