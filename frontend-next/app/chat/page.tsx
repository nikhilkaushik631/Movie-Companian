'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';

type ChatMessage = { 
  role: 'user' | 'bot'; 
  content: string; 
  recommendation_cards?: any[];
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const demoMode = (process.env.NEXT_PUBLIC_DEMO || '0') === '1';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'bot',
      content:
        "Welcome to the Cinemizer Chatbot. Ask me about any title, get recommendations, or discover trending content!",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showTyping, setShowTyping] = useState(false);
  const examples = [
    'Recommend me action movies like John Wick',
    'What are the best sci-fi movies from 2024?',
    'Show me popular Bollywood movies from 2024',
    'I love psychological thrillers, suggest something',
    'Find me a romantic comedy to watch tonight'
  ];
  const [token, setToken] = useState<string | null>(null);
  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginMessage, setLoginMessage] = useState('');
  const [showLogin, setShowLogin] = useState(false);
  const [sessions, setSessions] = useState<any[]>([]);
  const [loadingChats, setLoadingChats] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsItem, setDetailsItem] = useState<any>(null);
  const [detailsLLM, setDetailsLLM] = useState('');
  const sessionIdRef = useRef<string>('');

  const startNewChat = () => {
    if (messages.length > 1) {
      if (confirm('Are you sure you want to start a new chat? This will clear the current conversation.')) {
        setMessages([
          {
            role: 'bot',
            content: "Welcome to the Cinemizer Chatbot. Ask me about any title, get recommendations, or discover trending content!",
          },
        ]);
        sessionIdRef.current = crypto.randomUUID();
      }
    } else {
      // If only welcome message, no need to confirm
      setMessages([
        {
          role: 'bot',
          content: "Welcome to the Cinemizer Chatbot. Ask me about any title, get recommendations, or discover trending content!",
        },
      ]);
      sessionIdRef.current = crypto.randomUUID();
    }
  };

  useEffect(() => {
    sessionIdRef.current = crypto.randomUUID();
    if (typeof window !== 'undefined') {
      const t = localStorage.getItem('cinemizer_token');
      setToken(t);
      // Temporarily disable login gating for testing
      setShowLogin(false);
      const prefill = localStorage.getItem('cinemizer_prefill');
      if (prefill && prefill.trim()){
        localStorage.removeItem('cinemizer_prefill');
        setTimeout(()=>{ send(prefill); }, 200);
      }
    }
  }, []);

  const authHeaders = () => (token ? { Authorization: `Bearer ${token}` } : {});

  const openCardDetails = async (item: any) => {
    try {
      const external_id = item.id?.toString() || '';
      const title = item.title || item.name || '';
      const year = item.release_date || item.first_air_date ? 
        new Date(item.release_date || item.first_air_date).getFullYear().toString() : '';
      const media_type = item.name ? 'tv' : 'movie';
      
      setDetailsItem(item);
      setDetailsOpen(true);
      
      const r = await fetch(`${apiBase}/summary/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          external_id, 
          title, 
          year, 
          media_type, 
          source: 'tmdb' 
        }),
      });
      
      if (r.ok) {
        const j = await r.json();
        setDetailsLLM(j.summary);
      } else {
        setDetailsLLM('Could not retrieve summary at this time.');
      }
    } catch {
      setDetailsLLM('Could not retrieve summary at this time.');
    }
  };

  const send = async (text: string) => {
    if (!text.trim()) return;
    setMessages((m) => [...m, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);
    setShowTyping(true);
    try {
      if (demoMode) {
        const fake = `Demo response about: ${text}`;
        await new Promise((r) => setTimeout(r, 400));
        setMessages((m) => [...m, { role: 'bot', content: fake }]);
        return;
      }
      // Use demo endpoint for now (no authentication required)
      const r = await fetch(`${apiBase}/chat/demo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sessionIdRef.current }),
      });
      if (!r.ok) {
        const t = await r.text();
        setMessages((m) => [...m, { role: 'bot', content: `Error: ${r.status} - ${t}` }]);
      } else {
        const j = await r.json();
        const botMessage: ChatMessage = {
          role: 'bot',
          content: j.response,
          recommendation_cards: j.recommendation_cards || []
        };
        setMessages((m) => [...m, botMessage]);
        if (ttsEnabled) {
          try {
            const resp = await fetch(`${apiBase}/voice/tts`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: j.response })});
            if (resp.ok) {
              const blob = await resp.blob();
              const url = URL.createObjectURL(blob);
              const audio = new Audio(url);
              audio.play().catch(()=>{});
            }
          } catch {}
        }
      }
    } catch (e: any) {
      setMessages((m) => [...m, { role: 'bot', content: 'Network error. Is the API running?' }]);
    } finally {
      setLoading(false);
      setShowTyping(false);
    }
  };

  return (
    <div>
      <div className="container">
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', gap:8, flexWrap:'wrap' }}>
          <h2 style={{ marginTop: 0 }}>💬 Chatbot</h2>
          <button 
            className="btn btn-ghost" 
            onClick={startNewChat}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            ✨ New Chat
          </button>
        </div>

        {showLogin ? (
          <div style={{ maxWidth: 420, margin: '20px auto', background: 'rgba(20,15,10,0.95)', border: '1px solid rgba(230,220,198,0.15)', borderRadius: 16, padding: 16 }}>
            <h3 style={{ color: '#e6dcc6', textAlign: 'center', margin: 0 }}>Sign in to continue</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
              <input value={loginEmail} onChange={(e) => setLoginEmail(e.target.value)} placeholder="Email" style={{ padding: '10px 14px', borderRadius: 8, border: '1px solid rgba(230,220,198,0.2)', background: 'rgba(230,220,198,0.08)', color: '#e6dcc6' }} />
              <input type="password" value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)} placeholder="Password" style={{ padding: '10px 14px', borderRadius: 8, border: '1px solid rgba(230,220,198,0.2)', background: 'rgba(230,220,198,0.08)', color: '#e6dcc6' }} />
            </div>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'center', marginTop: 10 }}>
              <button onClick={async ()=>{ setLoginMessage(''); if(!loginEmail||!loginPassword){setLoginMessage('Enter email and password'); return;} try{ if(demoMode){ const fake=`demo_${Date.now()}`; localStorage.setItem('cinemizer_token', fake); setToken(fake); setShowLogin(false); setLoginMessage('Signed in (demo).'); return;} const form=new URLSearchParams(); form.append('username',loginEmail); form.append('password',loginPassword); const r=await fetch(`${apiBase}/auth/token`,{method:'POST',body:form}); if(!r.ok){ setLoginMessage('Login failed'); return;} const j=await r.json(); localStorage.setItem('cinemizer_token', j.access_token); setToken(j.access_token); setShowLogin(false);}catch{ setLoginMessage('Network error'); } }} style={{ background: 'rgba(230,220,198,0.12)', color: '#e6dcc6', border: '1px solid rgba(230,220,198,0.3)', padding: '10px 14px', borderRadius: 10 }}>Sign In</button>
              <button onClick={async ()=>{ setLoginMessage(''); if(!loginEmail||!loginPassword){setLoginMessage('Enter email and password'); return;} try{ if(demoMode){ const fake=`demo_${Date.now()}`; localStorage.setItem('cinemizer_token', fake); setToken(fake); setShowLogin(false); setLoginMessage('Account created (demo).'); return;} const r=await fetch(`${apiBase}/auth/signup`,{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({email:loginEmail,password:loginPassword})}); if(!r.ok){ setLoginMessage('Signup failed'); return;} const j=await r.json(); localStorage.setItem('cinemizer_token', j.access_token); setToken(j.access_token); setShowLogin(false);}catch{ setLoginMessage('Network error'); } }} style={{ background: 'transparent', color: '#e6dcc6', border: '1px solid rgba(230,220,198,0.3)', padding: '10px 14px', borderRadius: 10 }}>Create Account</button>
            </div>
            {loginMessage && <div style={{ marginTop: 8, textAlign: 'center' }}>{loginMessage}</div>}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16 }}>
            <section className="card">
              <div className="chat-window">
                {messages.map((m, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: 12 }}>
                    <div className={`bubble ${m.role === 'user' ? 'user' : 'bot'}`}>
                      {m.role === 'bot' ? (
                        <div>
                          <ReactMarkdown
                            components={{
                              // Custom styling for markdown elements
                              h2: ({node, ...props}) => <h2 style={{margin: '12px 0 8px 0', fontSize: '18px', fontWeight: 'bold', color: 'inherit'}} {...props} />,
                              h3: ({node, ...props}) => <h3 style={{margin: '10px 0 6px 0', fontSize: '16px', fontWeight: 'bold', color: 'inherit'}} {...props} />,
                              strong: ({node, ...props}) => <strong style={{fontWeight: '700', color: 'inherit'}} {...props} />,
                              ol: ({node, ...props}) => <ol style={{margin: '8px 0', paddingLeft: '20px'}} {...props} />,
                              li: ({node, ...props}) => <li style={{margin: '4px 0', lineHeight: '1.4'}} {...props} />,
                              p: ({node, ...props}) => <p style={{margin: '6px 0', lineHeight: '1.4'}} {...props} />,
                            }}
                          >
                            {m.content}
                          </ReactMarkdown>
                          {m.recommendation_cards && m.recommendation_cards.length > 0 && (
                            <div style={{ marginTop: '16px' }}>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', maxWidth: '600px' }}>
                                {m.recommendation_cards.map((card: any, idx: number) => (
                                  <div
                                    key={idx}
                                    onClick={() => openCardDetails(card)}
                                    className="item"
                                    style={{
                                      cursor: 'pointer',
                                      background: 'linear-gradient(145deg, rgba(30,25,20,0.9), rgba(25,20,15,0.8))',
                                      border: '1px solid rgba(0,255,255,0.15)',
                                      borderRadius: '12px',
                                      overflow: 'hidden',
                                      transition: 'all 0.3s ease',
                                      position: 'relative'
                                    }}
                                  >
                                    {card.poster_path && (
                                      <img
                                        src={`https://image.tmdb.org/t/p/w200${card.poster_path}`}
                                        alt={card.title || card.name}
                                        style={{
                                          width: '100%',
                                          height: '160px',
                                          objectFit: 'cover',
                                          transition: 'transform 0.3s ease'
                                        }}
                                        onError={(e) => {
                                          e.currentTarget.style.display = 'none';
                                        }}
                                      />
                                    )}
                                    <div style={{ padding: '8px' }}>
                                      <div style={{
                                        fontSize: '12px',
                                        fontWeight: '700',
                                        lineHeight: '1.3',
                                        height: '32px',
                                        overflow: 'hidden',
                                        display: '-webkit-box',
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: 'vertical',
                                        color: '#e6dcc6'
                                      }}>
                                        {card.title || card.name}
                                      </div>
                                      {card.release_date && (
                                        <div style={{ fontSize: '10px', color: '#ccc', marginTop: '4px' }}>
                                          {new Date(card.release_date || card.first_air_date).getFullYear()}
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        m.content
                      )}
                    </div>
                  </div>
                ))}
                {showTyping && (
                  <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}>
                    <div className="bubble bot" style={{ display:'inline-flex', alignItems:'center', gap:8 }}>
                      <span>Thinking</span>
                      <span className="typing-indicator">
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                      </span>
                    </div>
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
                <div style={{ 
                  flex: 1, 
                  display: 'flex', 
                  alignItems: 'center',
                  background: 'rgba(230,220,198,0.05)', 
                  border: '1px solid rgba(230,220,198,0.2)', 
                  borderRadius: 8,
                  padding: '4px'
                }}>
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        send(input);
                      }
                    }}
                    placeholder="Ask about movies, recommendations, or a title..."
                    style={{ 
                      flex: 1, 
                      padding: '8px 10px', 
                      border: 'none',
                      background: 'transparent', 
                      color: '#e6dcc6',
                      outline: 'none'
                    }}
                  />
                  <button 
                    onClick={() => setTtsEnabled(v=>!v)}
                    title={`Voice ${ttsEnabled ? 'On' : 'Off'}`}
                    style={{
                      background: 'none',
                      border: 'none',
                      fontSize: '18px',
                      cursor: 'pointer',
                      padding: '4px 8px',
                      opacity: ttsEnabled ? 1 : 0.6,
                      transition: 'all 0.2s ease',
                      borderRadius: '4px'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(230,220,198,0.1)';
                      e.currentTarget.style.opacity = '1';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'none';
                      e.currentTarget.style.opacity = ttsEnabled ? '1' : '0.6';
                    }}
                  >
                    {ttsEnabled ? '🔊' : '🔇'}
                  </button>
                </div>
                <button 
                  onClick={() => send(input)} 
                  disabled={loading || !input.trim()} 
                  className="btn btn-primary"
                  style={{
                    minWidth: '60px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '4px',
                    opacity: (loading || !input.trim()) ? 0.5 : 1
                  }}
                >
                  {loading ? '⏳' : '🚀'}
                </button>
              </div>
            </section>
            <aside className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 className="section-title" style={{ margin: 0 }}>🗂️ My Chats</h3>
                <button onClick={async ()=>{ setLoadingChats(true); try{ const r=await fetch(`${apiBase}/sessions`,{ headers: token?{ Authorization: `Bearer ${token}` }: {} }); if(r.ok){ const j=await r.json(); setSessions(j.sessions||[]);} } finally { setLoadingChats(false);} }} className="btn btn-ghost">{loadingChats ? 'Loading...' : 'Refresh'}</button>
              </div>
              <div style={{ display:'flex', flexDirection:'column', gap:8, marginTop:10 }}>
                {examples.map(ex => (
                  <button key={ex} className="btn btn-ghost" onClick={()=> send(ex)} style={{ textAlign:'left' }}>{ex}</button>
                ))}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 10, maxHeight: 520, overflow: 'auto' }}>
                {sessions.length === 0 ? (
                  <div style={{ opacity: 0.8 }}>No chats yet.</div>
                ) : (
                  sessions.map((s) => (
                    <button key={s.session_id} onClick={async ()=>{ try{ const r=await fetch(`${apiBase}/sessions/${s.session_id}/messages`,{ headers: token?{ Authorization: `Bearer ${token}` }: {} }); if(r.ok){ const j=await r.json(); const msgs = (j.messages||[]) as Array<{role:'user'|'bot';content:string}>; setMessages([{role:'bot', content:'Reopened conversation.'}, ...msgs]); } } catch{} }} className="btn btn-ghost" style={{ textAlign: 'left' }}>
                      {(s.title || 'Session')} • {new Date(s.created_at).toLocaleString()}
                    </button>
                  ))
                )}
              </div>
            </aside>
          </div>
        )}
        
        {detailsOpen && detailsItem && (
          <div className="modal-backdrop" onClick={() => setDetailsOpen(false)}>
            <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
              <div style={{ display: 'flex', gap: '20px' }}>
                {detailsItem.poster_path && (
                  <img
                    src={`https://image.tmdb.org/t/p/w300${detailsItem.poster_path}`}
                    alt={detailsItem.title || detailsItem.name}
                    style={{ width: '200px', height: '300px', objectFit: 'cover', borderRadius: '12px' }}
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                    }}
                  />
                )}
                <div style={{ flex: 1 }}>
                  <h2 style={{ margin: '0 0 10px 0', fontSize: '24px' }}>
                    {detailsItem.title || detailsItem.name}
                  </h2>
                  {(detailsItem.release_date || detailsItem.first_air_date) && (
                    <p style={{ margin: '0 0 15px 0', opacity: 0.8 }}>
                      {new Date(detailsItem.release_date || detailsItem.first_air_date).getFullYear()} • {detailsItem.media_type === 'tv' ? 'TV Show' : 'Movie'}
                    </p>
                  )}
                  {detailsItem.overview && (
                    <div>
                      <h3 style={{ margin: '15px 0 8px 0', fontSize: '16px' }}>Overview</h3>
                      <p style={{ lineHeight: '1.5', opacity: 0.9 }}>{detailsItem.overview}</p>
                    </div>
                  )}
                  <div>
                    <h3 style={{ margin: '15px 0 8px 0', fontSize: '16px' }}>AI Summary</h3>
                    <div style={{ opacity: 0.9, lineHeight: '1.5' }}>
                      {detailsLLM ? (
                        <ReactMarkdown>{detailsLLM}</ReactMarkdown>
                      ) : (
                        <div style={{ opacity: 0.6 }}>Loading summary...</div>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => setDetailsOpen(false)}
                    style={{
                      marginTop: '20px',
                      padding: '10px 20px',
                      background: 'rgba(139,115,85,0.2)',
                      border: '1px solid rgba(139,115,85,0.5)',
                      borderRadius: '8px',
                      color: '#e6dcc6',
                      cursor: 'pointer'
                    }}
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


