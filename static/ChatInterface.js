const { useState, useEffect, useRef } = React;

const AutoResizingTextarea = ({ value, onChange, disabled, placeholder }) => {
  const textareaRef = useRef(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 40), 200)}px`;
    }
  }, [value]);

  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none min-h-[40px] max-h-[200px]"
      rows={1}
    />
  );
};

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const sessionId = useRef(new Date().getTime().toString());
  const messagesEndRef = useRef(null);
  const [ragEnabled, setRagEnabled] = useState(true);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Trigger highlight.js after content updates
    document.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightBlock(block);
    });
  }, [messages]);

  useEffect(() => {
    console.log('Initializing WebSocket connection...');
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/${sessionId.current}`);

    wsRef.current.onopen = () => {
      console.log('Connected to server');
      setIsConnected(true);
      setError(null);
    };

    wsRef.current.onclose = () => {
      console.log('Disconnected from server');
      setIsConnected(false);
      setError('Connection lost');
      setIsLoading(false);
    };

    wsRef.current.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError('WebSocket error occurred');
      setIsLoading(false);
    };

    wsRef.current.onmessage = (event) => {
      console.log('Received message:', event.data);
      const data = JSON.parse(event.data);

      if (data.type === 'init') {
        setIsConnected(true);
        setError(null);
        setRagEnabled(data.rag_enabled);
        setMessages(msgs => [...msgs, {
          role: 'system',
          content: data.content,
          timestamp: data.timestamp
        }]);
        setIsLoading(false);
      } else if (data.type === 'response') {
        setMessages(msgs => [...msgs, {
          role: 'assistant',
          content: data.content,
          timestamp: data.timestamp
        }]);
        setIsLoading(false);
      } else if (data.type === 'error') {
        setError(data.content);
        setIsLoading(false);
      } else if (data.type === 'system') {
        setMessages(msgs => [...msgs, {
          role: 'system',
          content: data.content,
          timestamp: data.timestamp
        }]);
        setIsLoading(false);
      } else if (data.type === 'rag_status') {
        setRagEnabled(data.enabled);
        setMessages(msgs => [...msgs, {
          role: 'system',
          content: data.content,
          timestamp: data.timestamp
        }]);
        setIsLoading(false);
      } else if (data.type === 'session_loaded') {
        console.log('Loading session messages:', data.messages);
        // First add a system message about the loaded session
        const systemMessage = {
          role: 'system',
          content: `Loaded session ${data.session_info?.session_id || 'unknown'}`,
          timestamp: data.timestamp
        };

        // Then add all the loaded messages
        const loadedMessages = data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp
        }));

        // Update the messages state with system message and loaded messages
        setMessages([systemMessage, ...loadedMessages]);
        setIsLoading(false);
        setError(null);
      }
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleCommand = (command) => {
    if (!isConnected || isLoading) return;

    console.log('Sending command:', command);
    wsRef.current.send(JSON.stringify({
      type: 'command',
      command: command
    }));

    setIsLoading(true);
    setError(null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || !isConnected || isLoading) return;

    console.log('Sending message:', input);
    const newMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(msgs => [...msgs, newMessage]);

    wsRef.current.send(JSON.stringify({
      type: 'message',
      content: input
    }));

    setInput('');
    setIsLoading(true);
    setError(null);
  };

  const formatContent = (content) => {
    // First, safely extract and store think sections
    const thinkSections = [];
    let formattedContent = content.replace(/<think>\n?([\s\S]*?)<\/think>/g, (match, thinkContent) => {
      thinkSections.push(thinkContent.trim());
      return `[[THINK_SECTION_${thinkSections.length - 1}]]`;
    });

    // Configure marked options
    marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false,
      highlight: function (code, language) {
        const validLanguage = hljs.getLanguage(language) ? language : 'plaintext';
        return hljs.highlight(code, { language: validLanguage }).value;
      }
    });

    // Process markdown
    formattedContent = marked.parse(formattedContent);

    // Replace placeholders with formatted think sections
    formattedContent = formattedContent.replace(/\[\[THINK_SECTION_(\d+)\]\]/g, (match, index) => {
      const thinkContent = marked.parse(thinkSections[index]); // Process markdown inside think sections too
      return `
            <div class="bg-yellow-50 p-4 my-4 rounded-lg border-l-4 border-yellow-500">
                <div class="font-semibold text-yellow-800 mb-2">Thinking Process:</div>
                <div class="text-yellow-900">${thinkContent}</div>
            </div>
        `;
    });

    // Add styling for code blocks
    formattedContent = formattedContent.replace(
      /<pre><code class="language-(\w+)">/g,
      '<pre class="code-block"><code class="language-$1">'
    );
    formattedContent = formattedContent.replace(
      /<pre><code>/g,
      '<pre class="code-block"><code class="language-plaintext">'
    );

    // Style inline code
    formattedContent = formattedContent.replace(
      /<code>([^<]+)<\/code>/g,
      '<code class="inline-code">$1</code>'
    );

    return { __html: formattedContent };
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-6xl mx-auto bg-white rounded-lg shadow">
        {/* Status bar */}
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <span className={`w-3 h-3 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm text-gray-600">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center">
                <span className={`w-3 h-3 rounded-full mr-2 ${ragEnabled ? 'bg-blue-500' : 'bg-gray-500'}`}></span>
                <span className="text-sm text-gray-600">
                  {ragEnabled ? 'Code Context' : 'Chat Only'}
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Session ID: {sessionId.current}
              </div>
            </div>
          </div>

          {/* Command buttons */}
          <div className="flex flex-wrap gap-2 mt-4">
            <button
              onClick={() => handleCommand('help')}
              className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm flex items-center"
              disabled={!isConnected || isLoading}
            >
              <svg className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                <circle cx="12" cy="17" r="1" /> {/* This adds the dot at the bottom */}
              </svg>
              Help
            </button>
            <button
              onClick={() => handleCommand('refresh')}
              className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm flex items-center"
              disabled={!isConnected || isLoading}
            >
              <svg className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M23 4v6h-6" />
                <path d="M1 20v-6h6" />
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
              </svg>
              Refresh DB
            </button>
            <button
              onClick={() => handleCommand('save')}
              className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm flex items-center"
              disabled={!isConnected || isLoading}
            >
              <svg className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                <polyline points="17 21 17 13 7 13 7 21" />
                <polyline points="7 3 7 8 15 8" />
              </svg>
              Save
            </button>
            <label
              className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm flex items-center cursor-pointer"
            >
              <input
                type="file"
                accept=".json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                      try {
                        const session = JSON.parse(event.target.result);
                        wsRef.current.send(JSON.stringify({
                          type: 'command',
                          command: 'load',
                          data: session
                        }));
                      } catch (error) {
                        setError('Invalid session file format');
                      }
                    };
                    reader.readAsText(file);
                  }
                  // Reset the input so the same file can be selected again
                  e.target.value = '';
                }}
                disabled={!isConnected || isLoading}
              />
              <svg className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Load
            </label>
            <button
              onClick={() => handleCommand('debug')}
              className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm flex items-center"
              disabled={!isConnected || isLoading}
            >
              <svg className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
              Debug
            </button>
            <button
              onClick={() => handleCommand('toggle_rag')}
              className={`px-3 py-1 rounded-lg text-sm flex items-center ${ragEnabled
                ? 'bg-blue-100 hover:bg-blue-200 text-blue-800'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
                }`}
              disabled={!isConnected || isLoading}
            >
              <svg
                className={`w-4 h-4 mr-1 ${ragEnabled ? 'text-blue-600' : 'text-gray-600'}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
              RAG Mode: {ragEnabled ? 'On' : 'Off'}
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="px-4 py-2 m-4 bg-red-50 text-red-700 rounded-lg">
            Error: {error}
          </div>
        )}

        {/* Messages */}
        <div className="p-4">
          <div className="space-y-4 mb-4 max-h-[calc(100vh-300px)] overflow-y-auto">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg ${message.role === 'user'
                  ? 'bg-blue-50 ml-0 mr-12'
                  : message.role === 'system'
                    ? 'bg-gray-100 mx-0'
                    : 'bg-gray-50 ml-12 mr-0'
                  }`}
              >
                <div className="font-medium capitalize mb-2 flex justify-between">
                  <span>{message.role}</span>
                  <span className="text-sm text-gray-500">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div
                  className="prose max-w-none message-content"
                  dangerouslySetInnerHTML={formatContent(message.content)}
                />
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input form */}
          <form onSubmit={handleSubmit} className="flex flex-col space-y-2">
            <AutoResizingTextarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={!isConnected || isLoading}
            />
            <button
              type="submit"
              className={`px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 self-end
                ${isConnected && !isLoading
                  ? 'bg-blue-500 text-white hover:bg-blue-600'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              disabled={!isConnected || isLoading}
            >
              {isLoading ? (
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<ChatInterface />);