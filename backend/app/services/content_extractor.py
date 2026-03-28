"""
Content Extractor Service
Handles extraction of content from various sources:
- Google Docs (via export)
- News articles
- Generic web pages
- Direct content parsing
"""

import requests
import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
from ..utils.logger import get_logger

logger = get_logger('mirofish.content_extractor')


class ContentExtractor:
    """Service for extracting content from various sources"""
    
    # Google Docs constants
    GOOGLE_DOCS_URL_PATTERN = r'docs\.google\.com/document/d/([a-zA-Z0-9-_]+)'
    GOOGLE_DOCS_EXPORT_FORMAT = 'https://docs.google.com/document/d/{}/export?format=txt'
    
    # User Agent to avoid blocking
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    @staticmethod
    def extract_google_doc_id(url: str) -> Optional[str]:
        """Extract Google Docs document ID from URL"""
        try:
            match = re.search(ContentExtractor.GOOGLE_DOCS_URL_PATTERN, url)
            if match:
                return match.group(1)
            return None
        except Exception as e:
            logger.error(f"Failed to extract Google Docs ID: {e}")
            return None
    
    @staticmethod
    def fetch_google_doc_content(url: str) -> Dict[str, Any]:
        """
        Fetch content from Google Docs
        Returns: {'success': bool, 'content': str, 'error': str (optional)}
        """
        try:
            # Extract document ID
            doc_id = ContentExtractor.extract_google_doc_id(url)
            if not doc_id:
                return {
                    'success': False,
                    'error': 'Invalid Google Docs URL. Please provide a valid shareable link.'
                }
            
            # Construct export URL
            export_url = ContentExtractor.GOOGLE_DOCS_EXPORT_FORMAT.format(doc_id)
            
            # Fetch content
            response = requests.get(
                export_url,
                headers=ContentExtractor.DEFAULT_HEADERS,
                timeout=30
            )
            response.raise_for_status()
            
            content = response.text.strip()
            if not content:
                return {
                    'success': False,
                    'error': 'Google Docs document appears to be empty or inaccessible'
                }
            
            return {
                'success': True,
                'content': content
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out. Please check the URL and try again.'
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {
                    'success': False,
                    'error': 'Document not found. Please check the URL or sharing permissions.'
                }
            return {
                'success': False,
                'error': f'HTTP Error: {e.response.status_code}'
            }
        except Exception as e:
            logger.error(f"Error fetching Google Docs: {e}")
            return {
                'success': False,
                'error': f'Failed to fetch Google Docs content: {str(e)}'
            }
    
    @staticmethod
    def fetch_news_article(url: str) -> Dict[str, Any]:
        """
        Fetch content from a news article URL
        Returns: {'success': bool, 'content': str, 'title': str, 'error': str (optional)}
        """
        try:
            response = requests.get(
                url,
                headers=ContentExtractor.DEFAULT_HEADERS,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Try to extract content using basic HTML parsing
            content = ContentExtractor._extract_text_from_html(response.text)
            
            if not content or len(content.strip()) < 100:
                return {
                    'success': False,
                    'error': 'Could not extract meaningful content from the article'
                }
            
            return {
                'success': True,
                'content': content
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out. Please check the URL and try again.'
            }
        except requests.exceptions.HTTPError as e:
            return {
                'success': False,
                'error': f'HTTP Error: {e.response.status_code}'
            }
        except Exception as e:
            logger.error(f"Error fetching news article: {e}")
            return {
                'success': False,
                'error': f'Failed to fetch article content: {str(e)}'
            }
    
    @staticmethod
    def scrape_web_content(url: str) -> Dict[str, Any]:
        """
        Scrape content from any web page
        Returns: {'success': bool, 'content': str, 'error': str (optional)}
        """
        try:
            response = requests.get(
                url,
                headers=ContentExtractor.DEFAULT_HEADERS,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Extract text from HTML
            content = ContentExtractor._extract_text_from_html(response.text)
            
            if not content or len(content.strip()) < 50:
                return {
                    'success': False,
                    'error': 'Could not extract meaningful content from the page'
                }
            
            return {
                'success': True,
                'content': content
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timed out. Please check the URL and try again.'
            }
        except requests.exceptions.HTTPError as e:
            return {
                'success': False,
                'error': f'HTTP Error: {e.response.status_code}'
            }
        except Exception as e:
            logger.error(f"Error scraping web content: {e}")
            return {
                'success': False,
                'error': f'Failed to scrape web content: {str(e)}'
            }
    
    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        """
        Extract text content from HTML
        Uses simple parsing to avoid BeautifulSoup dependency initially
        """
        try:
            # Try to import BeautifulSoup, fall back to regex if not available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "meta", "link"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
            except ImportError:
                # Fallback: use regex to remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', html)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
