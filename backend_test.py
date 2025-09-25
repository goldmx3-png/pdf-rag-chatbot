#!/usr/bin/env python3

import requests
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import tempfile
import os

class RAGChatbotAPITester:
    def __init__(self, base_url="https://docuchatbot.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None
        self.uploaded_document_id = None
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        result = {
            "test_name": name,
            "status": "PASSED" if success else "FAILED",
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{status} - {name}")
        if details:
            print(f"   Details: {details}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'} if not files else {}

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=60)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=60)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            success = response.status_code == expected_status
            
            try:
                response_data = response.json() if response.content else {}
            except:
                response_data = {"raw_response": response.text}

            details = f"Status: {response.status_code}, Response: {json.dumps(response_data, indent=2)[:200]}..."
            self.log_test(name, success, details)

            return success, response_data

        except requests.exceptions.Timeout:
            self.log_test(name, False, "Request timed out")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Error: {str(e)}")
            return False, {}

    def test_api_root(self):
        """Test API root endpoint"""
        return self.run_test("API Root", "GET", "", 200)

    def test_status_endpoints(self):
        """Test status check endpoints"""
        # Test POST status
        success1, response1 = self.run_test(
            "Create Status Check",
            "POST",
            "status",
            200,
            data={"client_name": "test_client"}
        )
        
        # Test GET status
        success2, response2 = self.run_test(
            "Get Status Checks",
            "GET", 
            "status",
            200
        )
        
        return success1 and success2

    def create_test_pdf(self):
        """Create a simple test PDF file"""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            # Create a temporary PDF file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            
            # Create PDF content
            c = canvas.Canvas(temp_file.name, pagesize=letter)
            c.drawString(100, 750, "Test Document for RAG Chatbot")
            c.drawString(100, 720, "This is a test PDF document containing sample text.")
            c.drawString(100, 690, "The document discusses artificial intelligence and machine learning.")
            c.drawString(100, 660, "RAG (Retrieval-Augmented Generation) is a powerful technique.")
            c.drawString(100, 630, "It combines information retrieval with text generation.")
            c.save()
            
            return temp_file.name
        except ImportError:
            # Fallback: create a simple text file and rename it (for testing purposes)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='w')
            temp_file.write("This is a simple test file for PDF upload testing.")
            temp_file.close()
            return temp_file.name

    def test_document_upload(self):
        """Test document upload functionality"""
        try:
            # Create test PDF
            pdf_path = self.create_test_pdf()
            
            with open(pdf_path, 'rb') as f:
                files = {'file': ('test_document.pdf', f, 'application/pdf')}
                success, response = self.run_test(
                    "Upload PDF Document",
                    "POST",
                    "documents/upload",
                    200,
                    files=files
                )
            
            # Clean up
            os.unlink(pdf_path)
            
            if success and 'document_id' in response:
                self.uploaded_document_id = response['document_id']
                return True
            return False
            
        except Exception as e:
            self.log_test("Upload PDF Document", False, f"Error creating test PDF: {str(e)}")
            return False

    def test_get_documents(self):
        """Test getting documents list"""
        success, response = self.run_test(
            "Get Documents List",
            "GET",
            "documents",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} documents")
            return True
        return False

    def test_chat_functionality(self):
        """Test chat with documents"""
        # Generate session ID
        self.session_id = f"test_session_{int(time.time())}"
        
        success, response = self.run_test(
            "Chat with Documents",
            "POST",
            "chat",
            200,
            data={
                "message": "What is this document about?",
                "session_id": self.session_id
            }
        )
        
        if success and 'response' in response:
            print(f"   Chat response: {response['response'][:100]}...")
            return True
        return False

    def test_chat_history(self):
        """Test chat history retrieval"""
        if not self.session_id:
            self.log_test("Get Chat History", False, "No session ID available")
            return False
            
        success, response = self.run_test(
            "Get Chat History",
            "GET",
            f"chat/history/{self.session_id}",
            200
        )
        
        return success

    def test_chat_sessions(self):
        """Test getting chat sessions"""
        success, response = self.run_test(
            "Get Chat Sessions",
            "GET",
            "chat/sessions",
            200
        )
        
        return success

    def test_delete_document(self):
        """Test document deletion"""
        if not self.uploaded_document_id:
            self.log_test("Delete Document", False, "No document ID available")
            return False
            
        success, response = self.run_test(
            "Delete Document",
            "DELETE",
            f"documents/{self.uploaded_document_id}",
            200
        )
        
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting RAG Chatbot API Tests")
        print(f"   Base URL: {self.base_url}")
        print("=" * 60)

        # Test API availability
        self.test_api_root()
        
        # Test status endpoints
        self.test_status_endpoints()
        
        # Test document operations
        self.test_document_upload()
        self.test_get_documents()
        
        # Test chat functionality (only if documents are available)
        if self.uploaded_document_id:
            # Wait a bit for document processing
            print("\n⏳ Waiting for document processing...")
            time.sleep(3)
            
            self.test_chat_functionality()
            self.test_chat_history()
            self.test_chat_sessions()
        
        # Test document deletion
        self.test_delete_document()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate < 70:
            print("⚠️  Warning: Low success rate detected")
        elif success_rate == 100:
            print("🎉 All tests passed!")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = RAGChatbotAPITester()
    
    try:
        success = tester.run_all_tests()
        
        # Save test results
        results_file = "/app/test_results_backend.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_tests": tester.tests_run,
                "passed_tests": tester.tests_passed,
                "success_rate": (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0,
                "test_results": tester.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())