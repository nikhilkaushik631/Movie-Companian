'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const demoMode = (process.env.NEXT_PUBLIC_DEMO || '0') === '1';

export default function LoginPage(){
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(()=>{
    // Temporarily disable login page by redirecting to chat immediately
    router.replace('/chat');
  }, [router]);

  const signIn = async () => {
    setMsg('');
    if (!email || !password){ setMsg('Enter email and password'); return; }
    setLoading(true);
    try{
      if (demoMode){
        const fake = `demo_${Date.now()}`;
        localStorage.setItem('cinemizer_token', fake);
        router.replace('/chat');
        return;
      }
      const form = new URLSearchParams();
      form.append('username', email);
      form.append('password', password);
      const r = await fetch(`${apiBase}/auth/token`, { method:'POST', body: form });
      if (!r.ok){ setMsg('Login failed'); return; }
      const j = await r.json();
      localStorage.setItem('cinemizer_token', j.access_token);
      router.replace('/chat');
    } catch {
      setMsg('Network error');
    } finally { setLoading(false); }
  };

  const signUp = async () => {
    setMsg('');
    if (!email || !password){ setMsg('Enter email and password'); return; }
    setLoading(true);
    try{
      if (demoMode){
        const fake = `demo_${Date.now()}`;
        localStorage.setItem('cinemizer_token', fake);
        router.replace('/chat');
        return;
      }
      const r = await fetch(`${apiBase}/auth/signup`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email, password })});
      if (!r.ok){ setMsg('Signup failed'); return; }
      const j = await r.json();
      localStorage.setItem('cinemizer_token', j.access_token);
      router.replace('/chat');
    } catch { setMsg('Network error'); }
    finally { setLoading(false); }
  };

  return (
    <div className="container" style={{ maxWidth: 480 }}>
      <div className="card" style={{ marginTop: 28 }}>
        <h2 style={{ marginTop: 0, textAlign:'center' }}>🔐 Sign in to Cinemizer</h2>
        <div style={{ display:'flex', flexDirection:'column', gap:10, marginTop:12 }}>
          <input value={email} onChange={(e)=>setEmail(e.target.value)} placeholder="Email" style={{ padding:'10px 14px', borderRadius:8, border:'1px solid rgba(230,220,198,0.2)', background:'rgba(230,220,198,0.08)', color:'#e6dcc6' }} />
          <input type="password" value={password} onChange={(e)=>setPassword(e.target.value)} placeholder="Password" style={{ padding:'10px 14px', borderRadius:8, border:'1px solid rgba(230,220,198,0.2)', background:'rgba(230,220,198,0.08)', color:'#e6dcc6' }} />
        </div>
        <div style={{ display:'flex', gap:10, justifyContent:'center', marginTop:12 }}>
          <button className="btn btn-primary" onClick={signIn} disabled={loading}>Sign In</button>
          <button className="btn btn-ghost" onClick={signUp} disabled={loading}>Create Account</button>
        </div>
        {msg && <div style={{ marginTop:8, textAlign:'center' }}>{msg}</div>}
      </div>
    </div>
  );
}


