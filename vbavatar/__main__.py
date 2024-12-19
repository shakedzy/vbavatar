import os
import json
from datetime import datetime
from argparse import ArgumentParser
from .logger import Logger
from .browser import Browser
from .google_news_reader import GoogleNewsReader


def run():
    parser = ArgumentParser()
    parser.add_argument('-s', '--scrolls', type=int, dest='scrolls', help='Number of mouse-scrolls to perform (non-negative integer, default = 1)', default=1, required=False)
    parser.add_argument('-o', '--output-file', type=str, default='', dest='output_file', help='Output filename (defaults to `output_[run-time].json`)', required=False)
    parser.add_argument('--debug', action='store_true', dest='debug', help='Turn on debug mode', required=False, default=False)
    args = parser.parse_args()

    logger = Logger()
    logger.set_level('DEBUG' if args.debug else 'INFO')

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    output_filename = args.output_file or f'output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    output_filename = os.path.join('outputs', output_filename)

    with Browser() as context:
        g = GoogleNewsReader(browser_context=context, debug=args.debug)
        news = g.get_news(scrolls=args.scrolls)
        with open(output_filename, 'w') as f:
            json.dump(news, f)

    logger.info(f'Done, exported articles to: {output_filename}')
