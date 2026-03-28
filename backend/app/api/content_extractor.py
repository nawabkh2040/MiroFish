"""
Content Extraction API Routes
Handles extraction of content from:
- Google Docs
- News articles
- Web pages
- Direct content
"""

from flask import request, jsonify
from . import content_extractor_bp
from ..services.content_extractor import ContentExtractor
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.content_extractor')


@content_extractor_bp.route('/extract-content', methods=['POST'])
def extract_content():
    """
    Extract content from various sources
    
    Request JSON:
    {
        "type": "google_docs" | "news" | "web_scrape",
        "url": "https://..."
    }
    
    Response:
    {
        "success": true,
        "content": "extracted text content...",
        "error": "error message if any"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        content_type = data.get('type', '').lower()
        url = data.get('url', '').strip()
        
        # Validate inputs
        if not content_type:
            return jsonify({
                'success': False,
                'error': 'Content type is required (google_docs, news, or web_scrape)'
            }), 400
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return jsonify({
                'success': False,
                'error': 'URL must start with http:// or https://'
            }), 400
        
        # Route to appropriate extractor
        if content_type == 'google_docs':
            result = ContentExtractor.fetch_google_doc_content(url)
        elif content_type == 'news':
            result = ContentExtractor.fetch_news_article(url)
        elif content_type == 'web_scrape':
            result = ContentExtractor.scrape_web_content(url)
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid content type: {content_type}. Use google_docs, news, or web_scrape'
            }), 400
        
        # Log the operation
        if result.get('success'):
            content_length = len(result.get('content', ''))
            logger.info(f"Successfully extracted {content_length} characters from {content_type}: {url[:50]}...")
        else:
            logger.warning(f"Failed to extract from {content_type}: {result.get('error')}")
        
        return jsonify(result), 200 if result.get('success') else 400
        
    except Exception as e:
        logger.error(f"Unexpected error in extract_content: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500
