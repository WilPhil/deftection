
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import logging
import json
import tempfile
from datetime import datetime
from werkzeug.datastructures import FileStorage
from io import BytesIO

# Import controllers from flask-ai (main base)
from controllers.detection_controller import DetectionController

# Import services from flask-ai (main base)
from services.detection_service import DetectionService

# Import security scanning components from pythonsec2
from controllers.image_security_controller import ImageSecurityController

class EnhancedAPIServer:
    """
    API Server: Flask-AI (main) + Security Scanner + Real-time Frame Processing
    Support both JSON and form-data requests with frame processing
    """

    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)

        self.host = host
        self.port = port

        # Setup logging
        self._setup_logging()

        # Initialize services (flask-ai main base with detection)
        self.detection_service = DetectionService()

        # Initialize controllers (flask-ai main base with frame processing)
        self.detection_controller = DetectionController(self.detection_service)

        # Initialize security scanner (from pythonsec2)
        self.security_controller = ImageSecurityController()

        # Setup routes
        self._setup_routes()

        print("API Server initialized (Flask-AI + Security Scanner + Real-time Frame Processing)")

    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        """Setup all API endpoints - with real-time frame processing"""


        # Health and system endpoints (flask-ai)
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return self.detection_controller.health_check()

        @self.app.route('/api/system/info', methods=['GET'])
        def system_info():
            return self.detection_controller.get_system_info()

        @self.app.route('/api/system/status', methods=['GET'])
        def system_status():
            return self.detection_controller.get_system_status()

        # Detection endpoints (flask-ai) -
        @self.app.route('/api/detection/image', methods=['POST'])
        def detect_image():
            return self.detection_controller.process_image(request)

        @self.app.route('/api/detection/frame', methods=['POST'])
        def detect_frame():
            """
             Real-time frame processing with same logic as combined endpoint
            """
            return self.detection_controller.process_frame(request)

        @self.app.route('/api/detection/batch', methods=['POST'])
        def detect_batch():
            return self.detection_controller.process_batch(request)

        # Configuration endpoints (flask-ai)
        @self.app.route('/api/config/thresholds', methods=['GET'])
        def get_thresholds():
            return self.detection_controller.get_detection_thresholds()

        @self.app.route('/api/config/thresholds', methods=['PUT'])
        def update_thresholds():
            return self.detection_controller.update_detection_thresholds(request)

        @self.app.route('/api/config/reset', methods=['PUT'])
        def reset_thresholds():
            return self.detection_controller.reset_detection_thresholds(request)

        @self.app.route('/api/detection/combined', methods=['POST'])
        def detect_combined():
            """
             Combined defect detection + security scan
             Proper image data transfer to security scan
            """
            try:
                self.logger.info(f"Combined detection request - Content-Type: {request.content_type}")

                # Get defect detection result first
                defect_result = self.detection_controller.process_image(request)

                # Extract JSON response from Flask response object
                if hasattr(defect_result, 'get_json'):
                    defect_data = defect_result.get_json()
                    defect_status_code = defect_result.status_code
                elif isinstance(defect_result, tuple):
                    defect_response, defect_status_code = defect_result
                    if hasattr(defect_response, 'get_json'):
                        defect_data = defect_response.get_json()
                    else:
                        defect_data = defect_response.json if hasattr(defect_response, 'json') else defect_response
                else:
                    defect_data = defect_result
                    defect_status_code = 200

                # Check if security scan requested -  for both content types
                is_scan_threat = True  # DEFAULT TRUE as requested

                if request.is_json or 'application/json' in str(request.content_type):
                    json_data = request.get_json()
                    if json_data:
                        is_scan_threat = json_data.get('is_scan_threat', True)  # Default TRUE
                elif request.form:
                    is_scan_threat = request.form.get('is_scan_threat', 'true').lower() == 'true'

                self.logger.info(f"Security scan requested: {is_scan_threat}")

                if is_scan_threat:
                    #  Create proper security scan request with image data
                    try:
                        # Create a new request object for security scan with proper image data
                        security_result = self._perform_security_scan_with_proper_data(request)

                        # Extract security response
                        if isinstance(security_result, tuple):
                            security_data, security_status_code = security_result
                        else:
                            security_data = security_result
                            security_status_code = 200

                        # Merge responses if both successful
                        if defect_status_code == 200 and security_status_code == 200:
                            # Ensure defect_data has the right structure
                            if not isinstance(defect_data, dict):
                                defect_data = {'data': defect_data} if defect_data else {'data': {}}

                            # Add security scan to defect result
                            defect_data['security_scan'] = security_data.get('data', security_data)
                            defect_data['combined_analysis'] = True
                            defect_data['timestamp'] = datetime.now().isoformat()

                            return jsonify(defect_data), 200
                        else:
                            # Return defect result with security error info
                            if not isinstance(defect_data, dict):
                                defect_data = {'data': defect_data} if defect_data else {'data': {}}

                            defect_data['security_scan'] = {
                                'status': 'error',
                                'error': 'Security scan failed',
                                'details': security_data if security_status_code != 200 else 'Unknown error'
                            }
                            defect_data['combined_analysis'] = True
                            defect_data['timestamp'] = datetime.now().isoformat()

                            return jsonify(defect_data), defect_status_code

                    except Exception as security_error:
                        self.logger.error(f"Security scan error: {security_error}")
                        # Return defect result with security error info
                        if not isinstance(defect_data, dict):
                            defect_data = {'data': defect_data} if defect_data else {'data': {}}

                        defect_data['security_scan'] = {
                            'status': 'error',
                            'error': f'Security scan failed: {str(security_error)}',
                            'details': str(security_error)
                        }
                        defect_data['combined_analysis'] = True
                        defect_data['timestamp'] = datetime.now().isoformat()

                        return jsonify(defect_data), defect_status_code
                else:
                    # Return only defect detection
                    if not isinstance(defect_data, dict):
                        defect_data = {'data': defect_data} if defect_data else {'data': {}}

                    defect_data['combined_analysis'] = False
                    defect_data['timestamp'] = datetime.now().isoformat()

                    return jsonify(defect_data), defect_status_code

            except Exception as e:
                self.logger.error(f"Combined detection error: {e}")
                return jsonify({
                    'status': 'error',
                    'error': f'Combined detection failed: {str(e)}',
                    'timestamp': datetime.now().isoformat(),
                    'combined_analysis': False
                }), 500

        # ===========================
        # SECURITY SCANNER ENDPOINTS -
        # ===========================

        @self.app.route('/api/security/scan', methods=['POST'])
        def security_scan():
            """
             Security scan endpoint (normal format)
            Supports both JSON (base64) and form-data (file upload)
            Parameter: is_full_scan (boolean) - from request
            """
            return self.security_controller.scan_image(request)

        @self.app.route('/api/security/scan/laravel', methods=['POST'])
        def security_scan_laravel():
            """
             Security scan endpoint (Laravel format)
            Supports both JSON (base64) and form-data (file upload)
            Parameter: is_full_scan (boolean) - from request
            """
            return self.security_controller.scan_image_laravel(request)

        # Security health endpoints
        @self.app.route('/api/security/health', methods=['GET'])
        def security_health():
            return self.security_controller.health_check()

        @self.app.route('/api/security/stats', methods=['GET'])
        def security_stats():
            return self.security_controller.get_scanner_stats()

        # ===========================
        # ERROR HANDLERS
        # ===========================

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested API endpoint does not exist',
                'timestamp': datetime.now().isoformat()
            }), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal server error',
                'message': 'An internal error occurred while processing the request',
                'timestamp': datetime.now().isoformat()
            }), 500

        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad request',
                'message': 'Invalid request data or parameters',
                'timestamp': datetime.now().isoformat()
            }), 400

        @self.app.errorhandler(413)
        def file_too_large(error):
            return jsonify({
                'error': 'File too large',
                'message': 'File exceeds maximum allowed size',
                'timestamp': datetime.now().isoformat()
            }), 413

        @self.app.errorhandler(415)
        def unsupported_media_type(error):
            return jsonify({
                'error': 'Unsupported Media Type',
                'message': 'Content-Type not supported. Use application/json or multipart/form-data',
                'supported_types': ['application/json', 'multipart/form-data'],
                'timestamp': datetime.now().isoformat()
            }), 415

    def _perform_security_scan_with_proper_data(self, original_request):
        """FIXED: Perform security scan with proper image data transfer - NO MOCK"""
        try:
            # Extract image data from the original request
            image_data = None
            filename = None

            self.logger.info(f"Security scan - extracting data from request type: {original_request.content_type}")
            self.logger.info(f"Security scan - is_json: {original_request.is_json}")
            self.logger.info(f"Security scan - files available: {list(original_request.files.keys()) if original_request.files else 'None'}")

            if original_request.is_json or 'application/json' in str(original_request.content_type):
                # JSON request
                json_data = original_request.get_json()
                if json_data:
                    self.logger.info(f"Security scan - JSON keys: {list(json_data.keys())}")

                    # Get base64 image data
                    image_base64 = (json_data.get('image_base64') or
                                  json_data.get('image') or
                                  json_data.get('file_base64') or
                                  json_data.get('data'))

                    filename = json_data.get('filename', 'security_scan.jpg')

                    if image_base64:
                        # Handle data URI format
                        if isinstance(image_base64, str) and ',' in image_base64:
                            image_base64 = image_base64.split(',')[1]

                        import base64
                        image_data = base64.b64decode(image_base64)
                        self.logger.info(f"Security scan - JSON image data decoded: {len(image_data)} bytes")
                    else:
                        self.logger.error("Security scan - No base64 image data found in JSON")

            elif original_request.files:
                # Form-data request
                self.logger.info("Security scan - Processing form-data request")

                for field_name in ['image', 'file', 'upload', 'data']:
                    if field_name in original_request.files:
                        file_obj = original_request.files[field_name]
                        if file_obj.filename != '':
                            self.logger.info(f"Security scan - Found file in field '{field_name}': {file_obj.filename}")

                            # Reset file pointer to beginning
                            file_obj.seek(0)
                            image_data = file_obj.read()
                            filename = file_obj.filename or 'security_scan.jpg'
                            # Reset file pointer again for potential reuse
                            file_obj.seek(0)

                            self.logger.info(f"Security scan - Form file data read: {len(image_data)} bytes")
                            break
                        else:
                            self.logger.warning(f"Security scan - Field '{field_name}' has empty filename")

                if not image_data:
                    self.logger.error(f"Security scan - No valid file found in form fields: {list(original_request.files.keys())}")
            else:
                self.logger.error("Security scan - Request is neither JSON nor form-data")

            if not image_data:
                self.logger.error("Security scan - No image data extracted from request")
                return {
                    'status': 'error',
                    'data': {
                        'error_code': 'E002',
                        'message': 'No image data found for security scan',
                        'details': {'error': 'No image data available'},
                        'status': 'error'
                    }
                }, 400

            self.logger.info(f"Security scan - Successfully extracted: {len(image_data)} bytes, filename: {filename}")

            # FIXED: Direct call to security controller with proper data
            self.logger.info("Security scan - Calling security controller directly with extracted data")

            # Create proper file-like object for security controller
            from io import BytesIO
            from werkzeug.datastructures import FileStorage

            # Create file storage object that mimics uploaded file
            file_stream = BytesIO(image_data)
            file_storage = FileStorage(
                stream=file_stream,
                filename=filename,
                content_type='image/jpeg'
            )

            # Create request-like object for security controller
            class DirectSecurityRequest:
                def __init__(self, file_storage, filename):
                    self.files = {'image': file_storage}
                    self.form = {'is_full_scan': 'false'}  # Light scan for combined
                    self.content_type = 'multipart/form-data'
                    self.is_json = False
                    self.method = 'POST'

                def get_json(self):
                    return None

            direct_request = DirectSecurityRequest(file_storage, filename)

            self.logger.info("Security scan - Created direct request, calling security controller")

            # Perform security scan using direct request
            result = self.security_controller.scan_image_laravel(direct_request)

            self.logger.info(f"Security scan - Controller returned: {type(result)}")

            return result

        except Exception as e:
            self.logger.error(f"Error in security scan with proper data: {e}")
            import traceback
            self.logger.error(f"Security scan traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'data': {
                    'error_code': 'E999',
                    'message': f'Security scan setup failed: {str(e)}',
                    'details': {'error': str(e)},
                    'status': 'error'
                }
            }, 500

    def run(self, debug=False):
        """Start the API server"""
        print("Starting API Server (Flask-AI + Security Scanner + Real-time Frame Processing)")
        print("=" * 85)
        print(f"Server URL: http://{self.host}:{self.port}")
        print()
        print("FLASK-AI ENDPOINTS (Main Base) - ")
        print(f"  Health Check: GET /api/health")
        print(f"  System Info:  GET /api/system/info")
        print(f"  Detect Image: POST /api/detection/image")
        print(f"                Support: JSON (image_base64) + Form-data (image/file)")
        print(f"  Detect Frame: POST /api/detection/frame []")
        print(f"                Support: JSON (frame_base64/image_base64) + Form-data (frame/image/file)")
        print(f"                 Same logic as combined endpoint, optimized for real-time")
        print(f"  Batch Detect: POST /api/detection/batch")
        print(f"                Support: JSON (images array) + Form-data (multiple files)")
        print()
        print("COMBINED ENDPOINT - ")
        print(f"  Combined:     POST /api/detection/combined")
        print(f"                Support: JSON + Form-data")
        print(f"                Param: is_scan_threat=true for security scan (DEFAULT: TRUE)")
        print(f"                 Security scan data transfer")
        print(f"                 Proper defect type detection")
        print()
        print("SECURITY SCANNER ENDPOINTS - ")
        print(f"  Security Scan: POST /api/security/scan")
        print(f"                 Support: JSON (image_base64/image/file_base64/data) + Form-data")
        print(f"                 Param: is_full_scan=true/false")
        print(f"  Laravel Format: POST /api/security/scan/laravel")
        print(f"                  Support: JSON (image_base64/image/file_base64/data) + Form-data")
        print(f"                  Param: is_full_scan=true/false")
        print(f"  Security Health: GET /api/security/health")
        print(f"  Security Stats:  GET /api/security/stats")
        print("=" * 85)
        print("REAL-TIME FRAME PROCESSING FEATURES:")
        print("  - Same detection logic as combined endpoint")
        print("  - Smart processing with adaptive thresholds")
        print("  - Guaranteed defect detection")
        print("  - Frame-specific optimizations and caching")
        print("  - OpenAI analysis integration for real-time")
        print("  - Intelligent filtering and NMS")
        print("  - Confidence boosting for critical defects")
        print("=" * 85)
        print("ALL ENDPOINTS NOW SUPPORT BOTH JSON AND FORM-DATA!")
        print("JSON: Use 'image_base64', 'frame_base64' fields")
        print("Form-data: Use 'image', 'file', 'frame' fields")
        print(" Real-time frame processing with same quality as combined endpoint")
        print(" Frame-specific optimizations for real-time performance")
        print("=" * 85)

        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True, use_reloader=False)


def create_enhanced_api_server(host='0.0.0.0', port=5001):
    """Factory function to create API server"""
    return EnhancedAPIServer(host=host, port=port)

server_instance = create_enhanced_api_server()
app = server_instance.app

if __name__ == "__main__":
    # Create and start the API server
    server_instance.run(debug=True)
