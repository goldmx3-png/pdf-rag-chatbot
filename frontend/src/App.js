import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Badge } from './components/ui/badge';
import { Alert, AlertDescription } from './components/ui/alert';
import { toast } from 'sonner';
import { Toaster } from './components/ui/sonner';
import { Separator } from './components/ui/separator';
import { Upload, MessageSquare, FileText, Trash2, Send, Bot, User, Loader2 } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Generate or restore session ID on first load
  useEffect(() => {
    const savedSessionId = localStorage.getItem('currentSessionId');
    if (savedSessionId) {
      setSessionId(savedSessionId);
    } else if (!sessionId) {
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);
      localStorage.setItem('currentSessionId', newSessionId);
    }
  }, []);

  // Load chat history when session ID changes
  useEffect(() => {
    if (sessionId) {
      loadChatHistory(sessionId);
    }
  }, [sessionId]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load documents on component mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const generateSessionId = () => {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadDocuments = async () => {
    try {
      const response = await axios.get(`${API}/documents`);
      setDocuments(response.data);
    } catch (error) {
      console.error('Error loading documents:', error);
      toast.error('Failed to load documents');
    }
  };

  const loadChatHistory = async (session) => {
    try {
      const response = await axios.get(`${API}/chat/history/${session}`);
      const history = response.data;
      
      // Convert backend chat history to frontend message format
      const chatMessages = [];
      history.forEach(item => {
        // Add user message
        chatMessages.push({
          type: 'user',
          content: item.message,
          timestamp: new Date(item.timestamp),
        });
        
        // Add bot response
        chatMessages.push({
          type: 'bot',
          content: item.response,
          source_documents: item.source_documents?.map(filename => ({ 
            filename: filename,
            relevance_score: '1.00'
          })) || [],
          timestamp: new Date(item.timestamp),
        });
      });
      
      setMessages(chatMessages);
    } catch (error) {
      console.error('Error loading chat history:', error);
      // Don't show error toast for missing history, just start fresh
      setMessages([]);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === 'application/pdf') {
        setSelectedFile(file);
        toast.success(`Selected: ${file.name}`);
      } else {
        toast.error('Please select a PDF file');
        event.target.value = '';
      }
    }
  };

  const uploadDocument = async () => {
    if (!selectedFile) {
      toast.error('Please select a PDF file first');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/documents/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      toast.success(`Document uploaded successfully! Created ${response.data.chunks_created} chunks.`);
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      await loadDocuments();
    } catch (error) {
      console.error('Error uploading document:', error);
      toast.error('Failed to upload document');
    } finally {
      setIsUploading(false);
    }
  };

  const deleteDocument = async (documentId, filename) => {
    try {
      await axios.delete(`${API}/documents/${documentId}`);
      toast.success(`Deleted ${filename}`);
      await loadDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
      toast.error('Failed to delete document');
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    if (documents.length === 0) {
      toast.error('Please upload at least one PDF document before chatting');
      return;
    }

    const userMessage = currentMessage.trim();
    setCurrentMessage('');
    setIsLoading(true);

    // Add user message to chat
    const newUserMessage = {
      type: 'user',
      content: userMessage,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newUserMessage]);

    try {
      const response = await axios.post(`${API}/chat`, {
        message: userMessage,
        session_id: sessionId,
      });

      // Add bot response to chat
      const botMessage = {
        type: 'bot',
        content: response.data.response,
        source_documents: response.data.source_documents || [],
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message');
      
      // Add error message
      const errorMessage = {
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your message. Please try again.',
        timestamp: new Date(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewSession = () => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    localStorage.setItem('currentSessionId', newSessionId);
    setMessages([]);
    toast.success('New chat session started');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                  RAG Chatbot
                </h1>
                <p className="text-sm text-slate-600">Upload PDFs and chat with your documents</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button
                variant={activeTab === 'chat' ? 'default' : 'outline'}
                onClick={() => setActiveTab('chat')}
                className="flex items-center space-x-2"
                data-testid="chat-tab-btn"
              >
                <MessageSquare className="w-4 h-4" />
                <span>Chat</span>
              </Button>
              <Button
                variant={activeTab === 'documents' ? 'default' : 'outline'}
                onClick={() => setActiveTab('documents')}
                className="flex items-center space-x-2"
                data-testid="documents-tab-btn"
              >
                <FileText className="w-4 h-4" />
                <span>Documents</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'documents' && (
          <div className="space-y-6">
            {/* Upload Section */}
            <Card className="bg-white/60 backdrop-blur-sm border-slate-200" data-testid="upload-section">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Upload className="w-5 h-5" />
                  <span>Upload Documents</span>
                </CardTitle>
                <CardDescription>
                  Upload PDF documents to enable RAG-powered conversations
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center space-x-4">
                    <Input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      onChange={handleFileSelect}
                      className="flex-1"
                      data-testid="file-input"
                    />
                    <Button
                      onClick={uploadDocument}
                      disabled={!selectedFile || isUploading}
                      className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600"
                      data-testid="upload-btn"
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Uploading...
                        </>
                      ) : (
                        <>
                          <Upload className="w-4 h-4 mr-2" />
                          Upload PDF
                        </>
                      )}
                    </Button>
                  </div>
                  
                  {selectedFile && (
                    <Alert className="border-blue-200 bg-blue-50">
                      <AlertDescription>
                        Ready to upload: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Documents List */}
            <Card className="bg-white/60 backdrop-blur-sm border-slate-200" data-testid="documents-list">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>Uploaded Documents ({documents.length})</span>
                </CardTitle>
                <CardDescription>
                  Manage your uploaded PDF documents
                </CardDescription>
              </CardHeader>
              <CardContent>
                {documents.length === 0 ? (
                  <div className="text-center py-8 text-slate-500">
                    <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No documents uploaded yet</p>
                    <p className="text-sm">Upload a PDF to get started</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="flex items-center justify-between p-4 bg-white rounded-lg border border-slate-200"
                        data-testid={`document-${doc.id}`}
                      >
                        <div className="flex-1">
                          <h3 className="font-medium text-slate-900">{doc.filename}</h3>
                          <div className="flex items-center space-x-4 mt-1">
                            <Badge variant="secondary" className="text-xs">
                              {doc.chunks_count} chunks
                            </Badge>
                            <span className="text-xs text-slate-500">
                              {new Date(doc.upload_time).toLocaleDateString()}
                            </span>
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteDocument(doc.id, doc.filename)}
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                          data-testid={`delete-${doc.id}`}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="space-y-6">
            {/* Chat Header */}
            <Card className="bg-white/60 backdrop-blur-sm border-slate-200">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-slate-900">Chat with Documents</h2>
                    <p className="text-sm text-slate-600">
                      {documents.length} document{documents.length !== 1 ? 's' : ''} available for questions
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    onClick={startNewSession}
                    className="flex items-center space-x-2"
                    data-testid="new-session-btn"
                  >
                    <MessageSquare className="w-4 h-4" />
                    <span>New Session</span>
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Chat Messages */}
            <Card className="bg-white/60 backdrop-blur-sm border-slate-200" data-testid="chat-container">
              <CardContent className="p-0">
                <div className="h-96 overflow-y-auto p-4 space-y-4">
                  {messages.length === 0 ? (
                    <div className="text-center py-8 text-slate-500">
                      <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="font-medium">Ready to answer your questions</p>
                      <p className="text-sm">
                        {documents.length > 0 
                          ? "Ask me anything about your uploaded documents"
                          : "Upload some PDF documents first to get started"}
                      </p>
                    </div>
                  ) : (
                    messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        data-testid={`message-${index}`}
                      >
                        <div
                          className={`max-w-xs md:max-w-md lg:max-w-lg xl:max-w-xl rounded-lg px-4 py-3 ${
                            message.type === 'user'
                              ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white'
                              : message.isError
                              ? 'bg-red-50 border border-red-200 text-red-800'
                              : 'bg-white border border-slate-200 text-slate-900'
                          }`}
                        >
                          <div className="flex items-start space-x-2">
                            {message.type === 'bot' && (
                              <Bot className="w-5 h-5 mt-0.5 text-blue-500" />
                            )}
                            {message.type === 'user' && (
                              <User className="w-5 h-5 mt-0.5 text-white" />
                            )}
                            <div className="flex-1">
                              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                {message.content}
                              </p>
                              
                              {message.source_documents && message.source_documents.length > 0 && (
                                <div className="mt-3 pt-3 border-t border-slate-200">
                                  <p className="text-xs text-slate-600 mb-2">Sources:</p>
                                  <div className="space-y-1">
                                    {message.source_documents.map((doc, idx) => (
                                      <Badge key={idx} variant="outline" className="text-xs mr-2">
                                        {doc.filename} ({doc.relevance_score})
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                  
                  {isLoading && (
                    <div className="flex justify-start" data-testid="loading-message">
                      <div className="bg-white border border-slate-200 rounded-lg px-4 py-3">
                        <div className="flex items-center space-x-2">
                          <Bot className="w-5 h-5 text-blue-500" />
                          <div className="flex items-center space-x-1">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span className="text-sm text-slate-600">Thinking...</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>

                <Separator />
                
                {/* Message Input */}
                <form onSubmit={sendMessage} className="p-4">
                  <div className="flex space-x-2">
                    <Textarea
                      value={currentMessage}
                      onChange={(e) => setCurrentMessage(e.target.value)}
                      placeholder="Ask a question about your documents..."
                      className="flex-1 min-h-0 resize-none"
                      rows={1}
                      disabled={isLoading || documents.length === 0}
                      data-testid="message-input"
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          sendMessage(e);
                        }
                      }}
                    />
                    <Button
                      type="submit"
                      disabled={!currentMessage.trim() || isLoading || documents.length === 0}
                      className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600"
                      data-testid="send-btn"
                    >
                      <Send className="w-4 h-4" />
                    </Button>
                  </div>
                  {documents.length === 0 && (
                    <p className="text-xs text-slate-500 mt-2">
                      Upload documents first to enable chat functionality
                    </p>
                  )}
                </form>
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;