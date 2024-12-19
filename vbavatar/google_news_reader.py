import os
import re
import json
import random
import ollama
import numpy as np
from tqdm import tqdm
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from PIL import Image, ImageDraw
from PIL.Image import Image as PillowImage
from time import sleep
from pydantic import BaseModel, field_validator
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import cast
from .browser import BrowserContext
from .logger import Logger
from .utils import dedent, torch_device, domain_of_url
from .news_page_scraper import NewsPageScraper
from .cache import Cache
from .types_ import Article


class Titles(BaseModel):
    titles: list[str]

    @field_validator("titles")
    def check_titles(cls, value: list[str], _) -> list[str]:
        if 'TITLE_1' in value:
            raise ValueError('TITLE_1 and TITLE_2 are example names, they are not the actual titles you should return!')
        else:
            return value


class GoogleNewsReader:
    SECTION_IMAGE_NAME_TEMPLATE = 'gn_S{i}.png'

    def __init__(self, 
                 *, 
                 browser_context: BrowserContext,
                 debug: bool = False
                 ):
        self._browser_context = browser_context
        self._debug = debug
        self.logger = Logger()
        self.scraper = NewsPageScraper()
        self.cache = Cache()
        self._torch_device = torch_device()
        self.florence = {
            'model': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).eval().to(self._torch_device),
            'processor': AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
        }
        self.urls = {
            # Google News -> Technology -> AI
            'AI': 'https://news.google.com/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRGRqTVhZU0JXVnVMVWRDR2dKSlRDZ0FQAQ/sections/CAQiQ0NCQVNMQW9JTDIwdk1EZGpNWFlTQW1WdUdnSlZVeUlOQ0FRYUNRb0hMMjB2TUcxcmVpb0pFZ2N2YlM4d2JXdDZLQUEqKggAKiYICiIgQ0JBU0Vnb0lMMjB2TURkak1YWVNBbVZ1R2dKVlV5Z0FQAVAB?hl=en-US&gl=US&ceid=US%3Aen'
        }

    def _draw_ocr_bboxes(self, image: PillowImage, img_path: str, prediction: dict) -> None:
        colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
                    'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
        draw = ImageDraw.Draw(image)
        bboxes, labels = prediction['quad_boxes'], prediction['labels']
        for box, label in zip(bboxes, labels):
            color = random.choice(colormap)
            new_box = np.array(box).tolist()
            draw.polygon(new_box, width=3, outline=color)                                           # pyright: ignore[reportArgumentType]
            draw.text((new_box[0]+8, new_box[1]+2), "{}".format(label), align="right", fill=color)  # pyright: ignore[reportOperatorIssue, reportIndexIssue]
        file_name = f"DEBUG_google_news_{img_path.split('/')[-1].split('.')[0]}.png"
        image.save(os.path.join(self.cache.directory, file_name))
        self.logger.debug(f"Saved: {file_name}") 
        
    def _ocr_and_bounding_boxes(self, img_path: str) -> list[tuple[str, list[float]]]:
        prompt = '<OCR_WITH_REGION>'
        image = Image.open(img_path).convert('RGB')
        model = self.florence['model']
        processor = self.florence['processor']
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(self._torch_device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].to(self._torch_device),
            pixel_values=inputs["pixel_values"].to(self._torch_device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer: dict = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        output: list[tuple[str, list[float]]] = []
        for text, bbox in zip(parsed_answer[prompt]['labels'], parsed_answer[prompt]['quad_boxes']):
            text = cast(str, text).strip()
            bbox = cast(list[float], bbox)  # [x1, y1, x2, y2, x3, y3, x4, y4]
            if text.startswith('</s>'): text.replace('</s>','')
            output.append((text, bbox))
        
        if self._debug:
            self._draw_ocr_bboxes(image, img_path, parsed_answer[prompt])

        return output
    
    def _bbox_center(self, 
                     coords: list[float] # [x1, y1, x2, y2, x3, y3, x4, y4]
                     ) -> tuple[int, int]:
        x_coords = coords[0::2]  # Extract x coordinates (every 2nd value starting from 0)
        y_coords = coords[1::2]  # Extract y coordinates (every 2nd value starting from 1)
        center_x = sum(x_coords) / 4  # Average of x coordinates
        center_y = sum(y_coords) / 4  # Average of y coordinates
        return int(round(center_x)), int(round(center_y))
    
    def _titles_from_image(self, img_path: str) -> list[str]:
        prompt = dedent(
            f"""
            ## Task
            You are provided with a partial-screenshot taken from a news website. The image provided
            should contain titles of articles mentioned in the new site.
            Your task is to locate these titles and return only the titles mentioned.
            If there's only one title, return a list with a single element.
            If there are no titles - return an empty list.

            ## Output Format
            Return the output in the following JSON format:
            ```json
            {{
                'titles': ["TITLE_1", "TITLE_2", ...]
            }}
            ```
            """)
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{'role': 'user', 'content': prompt, 'images': [img_path]}],
            options={'temperature': 0.3},
            format=Titles.model_json_schema(),
        )['message']['content']
        titles: list[str] = json.loads(response)['titles']
        return titles
    
    def _filter_ocr_texts_and_bboxes(self, texts_and_bboxes: list[tuple[str, list[float]]]) -> list[tuple[str, list[float]]]:
        # bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
        remaining: list[tuple[str, list[float]]] = []
        for text, bbox in texts_and_bboxes:
            highest_y = max(bbox[1::2])  # lowest y on screen
            words = [w for w in text.split(' ') if w]
            if len(words) > 2 and highest_y > 110:  # avoiding the Google News top bar
                remaining.append((text, bbox))
        self.logger.debug(f'Texts/BBoxes filter: {len(remaining)}/{len(texts_and_bboxes)} remaining')
        return remaining
    
    def _analyze_single_screenshot(self, img_path: str) -> list[tuple[int, int]]:
        with tqdm(total=2, desc=f'Analyzing image {img_path.split("/")[-1]}') as bar:
            def empty_response() -> list[tuple[int, int]]:
                bar.n = 2
                bar.refresh()
                return []
            
            try:
                titles = self._titles_from_image(img_path)
                bar.update()
                self.logger.debug(f'Section {img_path}: {len(titles)} titles from image = {titles}')
                if not titles: return empty_response()
                texts_and_bboxes = self._ocr_and_bounding_boxes(img_path)
                texts_and_bboxes = self._filter_ocr_texts_and_bboxes(texts_and_bboxes)
                bar.update()
                if not texts_and_bboxes: return empty_response()

                where_to_click: list[tuple[int, int]] = []
                for title in titles:
                    title = title.lower()
                    indices = [i for i, (text, bbox) in enumerate(texts_and_bboxes) if text.lower() in title]
                    if indices:
                        bbox = texts_and_bboxes[indices[0]][1]
                        center = self._bbox_center(bbox)
                        where_to_click.append(center)
                return where_to_click

            except Exception as e:
                self.logger.error(f'{e.__class__.__name__}: {e}', stack_lines=10)
                return empty_response()
        
    def _index_to_section_image_name(self, i: int) -> str:
        return self.SECTION_IMAGE_NAME_TEMPLATE.format(i=i)
    
    def _section_image_name_to_index(self, name: str) -> int:
        escaped_template = re.escape(self.SECTION_IMAGE_NAME_TEMPLATE).replace(r'\{i\}', r'(\d+)')
        match = re.fullmatch(escaped_template, name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("The input string does not match the given template.")

    def get_news(self,
                 *,
                 max_threads: int | None = 3,
                 scrolls: int = 6
                 ) -> dict[str, list[Article]]:
        output: dict[str, list[Article]] = {}
        for k, url in self.urls.items():
            self.logger.info(f'Visiting {k}: {url}')

            google_news_page = self._browser_context.new_page()
            google_news_page.goto(url)
            google_news_page.wait_for_load_state()
            height: int = google_news_page.viewport_size['height']  # pyright: ignore[reportOptionalSubscript]
            output[k] = []
            for scroll in range(scrolls):
                google_news_page.bring_to_front()
                if scroll > 0: 
                    google_news_page.mouse.wheel(0, height)
                    sleep(1)
                
                screenshot_path = os.path.join(self.cache.directory, self._index_to_section_image_name(scroll))
                google_news_page.screenshot(path=screenshot_path)
                self.logger.debug(f"Analyzing screenshot of scroll {scroll}: {screenshot_path}")

                points_to_click = self._analyze_single_screenshot(screenshot_path)
                self.logger.debug(f'Points to click: {points_to_click}')
                if not points_to_click:
                    self.logger.warning(f"Found no titles to click in screenshot: {screenshot_path}")
                    continue

                raw_pages: list[tuple[str, str]] = []
                not_google_news_url_re_pattern = re.compile(r"^(?!https?:\/\/(?:www\.)?news\.google\.com).*$")

                def url_not_from_google_news(page) -> bool:
                    nonlocal not_google_news_url_re_pattern
                    page.wait_for_url(not_google_news_url_re_pattern)
                    return 'news.google.com' not in domain_of_url(page.url)

                for x,y in points_to_click:
                    self.logger.debug(f'Clicking on ({x}, {y})')
                    try:
                        with self._browser_context.expect_page(predicate=url_not_from_google_news) as page_info:
                            google_news_page.mouse.click(x, y)
                    except PlaywrightTimeoutError as e:
                        self.logger.error(f'No page opened after clicking on ({x},{y})! | {e.__class__.__name__}: {e}')
                        continue
                    new_page = page_info.value
                    self.logger.debug(f'Opened new page: {new_page.url}')
                    if new_page.url not in [a['url'] for a in output[k]]:
                        new_page.bring_to_front()
                        html = new_page.content()
                        page_url = new_page.url
                        raw_pages.append((page_url, html))
                    else:
                        self.logger.debug('Already opened this URL, skipping')
                    google_news_page.bring_to_front()
                
                if not raw_pages:
                    self.logger.warning("No pages opened!")
                    continue

                u_list, h_list = tuple(zip(*raw_pages))
                u_list = cast(list[str], u_list)
                h_list = cast(list[str], h_list)
                articles: list[Article] = []
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    results = list(tqdm(executor.map(self.scraper.from_html, h_list, u_list), desc="Cleaning texts", total=len(u_list)))
                for result in results:
                    articles.append(result)
                self.logger.info(f'Extracted {len(articles)} articles from scroll {scroll}/{scrolls}')
                output[k].extend(articles)
        return output
    