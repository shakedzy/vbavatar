from playwright.sync_api import sync_playwright, BrowserContext, Playwright


class Browser:
    def __init__(self, *, height: int = 900, width: int = 1050) -> None:
        self._client = sync_playwright()
        self._browser = None
        self._context = None
        self._viewport = {'height': height, 'width': width}

    def _get_device(self, playwright: Playwright) -> dict:
        device = playwright.devices['Desktop Chrome']
        device['viewport'] = self._viewport
        return device

    def __enter__(self) -> BrowserContext:
        playwright = self._client.__enter__()
        browser_client = playwright.chromium
        self._browser = browser_client.launch(headless=False) 
        context = self._browser.new_context(**self._get_device(playwright))
        self._context = context
        return context
    
    def __exit__(self, *args) -> None:
        if self._browser: self._browser.close()
        if self._client: self._client.__exit__(*args)
