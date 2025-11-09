'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';

type ChatMessage = { role: 'user' | 'bot'; content: string };

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const demoMode = (process.env.NEXT_PUBLIC_DEMO || '0') === '1';
const tmdbImageBaseUrl = 'https://image.tmdb.org/t/p/w500';

export default function Page() {
  const router = useRouter();
  const sessionIdRef = useRef<string>('');
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [voiceApi, setVoiceApi] = useState(false);
  const [genres, setGenres] = useState<Record<string, string>>({});
  const [token, setToken] = useState<string | null>(null);
  // Home page no longer shows auth or chat UI

  // Tabs and trending results
  const [movieTab, setMovieTab] = useState<
    'trending_movies_day' | 'trending_movies_week' | 'popular_movies' | 'top_rated_movies' | 'now_playing' | 'upcoming_movies'
  >('trending_movies_day');
  const [tvTab, setTvTab] = useState<
    'trending_tv_day' | 'trending_tv_week' | 'popular_tv' | 'top_rated_tv' | 'airing_today' | 'on_the_air'
  >('trending_tv_day');
  const [indiaTab, setIndiaTab] = useState<'bollywood' | 'south_indian' | 'regional'>('bollywood');
  const [animeTab, setAnimeTab] = useState<'anime_top_rated' | 'anime_airing'>('anime_top_rated');
  const [movieResults, setMovieResults] = useState<any[]>([]);
  const [tvResults, setTvResults] = useState<any[]>([]);
  const [indiaResults, setIndiaResults] = useState<any[]>([]);
  const [animeResults, setAnimeResults] = useState<any[]>([]);
  const [loadingMovies, setLoadingMovies] = useState(true);
  const [loadingTV, setLoadingTV] = useState(true);
  const [loadingIndia, setLoadingIndia] = useState(true);
  const [loadingAnime, setLoadingAnime] = useState(true);
  const [sessions, setSessions] = useState<any[]>([]);
  const [loadingChats, setLoadingChats] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsItem, setDetailsItem] = useState<any | null>(null);
  const [detailsLLM, setDetailsLLM] = useState<string>('');
  // Home search
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMsg, setSearchMsg] = useState('');
  const [searchLoading, setSearchLoading] = useState(false);
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);

  // AI Summaries for trending cards and details
  const [aiSummaries, setAiSummaries] = useState<Record<string, string>>({});
  const [loadingSummaries, setLoadingSummaries] = useState<Set<string>>(new Set());
  // Cache for detail summaries (full summaries when clicking cards) with localStorage persistence
  const [detailSummaries, setDetailSummaries] = useState<Record<string, string>>({});

  // Debounce search query - wait 500ms after user stops typing
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery);
    }, 500);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Auto-search when debounced query changes (only if non-empty and different from last search)
  const lastSearchRef = useRef('');
  useEffect(() => {
    const trimmed = debouncedSearchQuery.trim();
    if (trimmed && trimmed !== lastSearchRef.current && trimmed.length >= 2) {
      lastSearchRef.current = trimmed;
      searchTMDBHome(trimmed);
    }
  }, [debouncedSearchQuery]);

  // Load cached summaries from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        // Cache version for invalidation - increment this to clear all caches
        const CACHE_VERSION = 'v2'; // Changed from v1 to clear old cache
        const currentVersion = localStorage.getItem('cinemizer_cache_version');

        // Clear old cache if version mismatch
        if (currentVersion !== CACHE_VERSION) {
          console.log('🔄 Clearing old cache and reloading summaries...');
          localStorage.removeItem('cinemizer_detail_summaries');
          localStorage.setItem('cinemizer_cache_version', CACHE_VERSION);
          setDetailSummaries({});
          return;
        }

        const cached = localStorage.getItem('cinemizer_detail_summaries');
        if (cached) {
          const parsed = JSON.parse(cached);
          // Check expiration (24 hours)
          const now = Date.now();
          const validCache: Record<string, string> = {};

          Object.entries(parsed).forEach(([key, value]: [string, any]) => {
            if (value.timestamp && (now - value.timestamp) < 24 * 60 * 60 * 1000) {
              validCache[key] = value.summary;
            }
          });

          if (Object.keys(validCache).length > 0) {
            console.log(`✅ Loaded ${Object.keys(validCache).length} cached summaries`);
            setDetailSummaries(validCache);
          }
        }
      } catch {}
    }
  }, []);

  // Save summaries to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined' && Object.keys(detailSummaries).length > 0) {
      try {
        const toCache: Record<string, any> = {};
        const now = Date.now();

        Object.entries(detailSummaries).forEach(([key, summary]) => {
          toCache[key] = {
            summary,
            timestamp: now
          };
        });

        localStorage.setItem('cinemizer_detail_summaries', JSON.stringify(toCache));
      } catch {}
    }
  }, [detailSummaries]);

  useEffect(() => {
    sessionIdRef.current = crypto.randomUUID();
    if (typeof window !== 'undefined') {
      const t = localStorage.getItem('cinemizer_token');
      setToken(t);
    }
  }, []);

  // Load movie genres (for labels)
  useEffect(() => {
    (async () => {
      if (demoMode) return;
      try {
        const url = new URL(`${apiBase}/tmdb/genre/movie/list`);
        const r = await fetch(url);
        if (r.ok) {
          const j = await r.json();
          const map: Record<string, string> = {};
          (j.genres || []).forEach((g: any) => (map[String(g.id)] = g.name));
          setGenres(map);
        }
      } catch {}
    })();
  }, []);

  // TMDB helper and loaders
  const authHeaders = (): Record<string, string> => (token ? { Authorization: `Bearer ${token}` } : {});
  const tmdbRequest = async (endpoint: string, params?: Record<string, string>) => {
    const url = new URL(`${apiBase}/tmdb${endpoint}`);
    if (params) Object.entries(params).forEach(([k, v]) => v && url.searchParams.append(k, v));
    const r = await fetch(url, { headers: authHeaders() });
    if (!r.ok) throw new Error(`TMDB ${endpoint} ${r.status}`);
    return r.json();
  };

  const loadMovies = async (tab: typeof movieTab) => {
    setLoadingMovies(true);
    if (demoMode) {
      const results = Array.from({ length: 5 }, (_, i) => ({ id: i, title: `Movie ${i + 1}`, poster_path: '', popularity: 100 - i, vote_average: 7.5 }));
      setMovieResults(results);
      setLoadingMovies(false);
      return;
    }
    let endpoint = '/trending/movie/day';
    if (tab === 'trending_movies_week') endpoint = '/trending/movie/week';
    if (tab === 'popular_movies') endpoint = '/movie/popular';
    if (tab === 'top_rated_movies') endpoint = '/movie/top_rated';
    if (tab === 'now_playing') endpoint = '/movie/now_playing';
    if (tab === 'upcoming_movies') endpoint = '/movie/upcoming';
    try{
      const data = await tmdbRequest(endpoint);
      const results = (data.results || []).slice(0, 5);
      setMovieResults(results);
      // Load AI summaries for the first few items
      if (!demoMode) {
        loadAISummariesForItems(results);
      }
    } catch {}
    finally { setLoadingMovies(false); }
  };

  const loadTV = async (tab: typeof tvTab) => {
    setLoadingTV(true);
    if (demoMode) {
      const results = Array.from({ length: 5 }, (_, i) => ({ id: i, name: `TV ${i + 1}`, poster_path: '', popularity: 100 - i, vote_average: 8.1 }));
      setTvResults(results);
      setLoadingTV(false);
      return;
    }
    let endpoint = '/trending/tv/day';
    if (tab === 'trending_tv_week') endpoint = '/trending/tv/week';
    if (tab === 'popular_tv') endpoint = '/tv/popular';
    if (tab === 'top_rated_tv') endpoint = '/tv/top_rated';
    if (tab === 'airing_today') endpoint = '/tv/airing_today';
    if (tab === 'on_the_air') endpoint = '/tv/on_the_air';
    try{
      const data = await tmdbRequest(endpoint);
      const results = (data.results || []).slice(0, 5);
      setTvResults(results);
      // Load AI summaries for the first few items
      if (!demoMode) {
        loadAISummariesForItems(results);
      }
    } catch {}
    finally { setLoadingTV(false); }
  };

  const loadAnime = async (tab: typeof animeTab) => {
    setLoadingAnime(true);
    if (demoMode) {
      const results = Array.from({ length: 6 }, (_, i) => ({ id: i, name: `Anime ${i + 1}`, poster_path: '', popularity: 90 - i, vote_average: 8.5 }));
      setAnimeResults(results);
      setLoadingAnime(false);
      return;
    }
    const params: Record<string, string> = { with_genres: '16', with_original_language: 'ja' };
    if (tab === 'anime_top_rated') {
      params.sort_by = 'vote_average.desc';
      (params as any)['vote_count.gte'] = '100';
    }
    if (tab === 'anime_airing') {
      params.sort_by = 'popularity.desc';
      (params as any)['air_date.gte'] = new Date().toISOString().split('T')[0];
    }
    try{
      const data = await tmdbRequest('/discover/tv', params);
      const results = (data.results || []).slice(0, 5);
      setAnimeResults(results);
      // Load AI summaries for the first few items
      if (!demoMode) {
        loadAISummariesForItems(results);
      }
    } catch {}
    finally { setLoadingAnime(false); }
  };

  const loadIndia = async (tab: typeof indiaTab) => {
    setLoadingIndia(true);
    if (demoMode) {
      const results = Array.from({ length: 5 }, (_, i) => ({ id: i, title: `IND ${i + 1}`, poster_path: '', popularity: 85 - i, vote_average: 7.9 }));
      setIndiaResults(results);
      setLoadingIndia(false);
      return;
    }
    const params: Record<string, string> = {};
    if (tab === 'bollywood') {
      params.with_original_language = 'hi';
      params.sort_by = 'popularity.desc';
    }
    if (tab === 'south_indian') {
      params.with_original_language = 'ta|te|ml|kn';
      params.sort_by = 'popularity.desc';
    }
    if (tab === 'regional') {
      params.with_original_language = 'bn|gu|mr|pa|or|as';
      params.sort_by = 'popularity.desc';
    }
    try{
      const data = await tmdbRequest('/discover/movie', params);
      const results = (data.results || []).slice(0, 5);
      setIndiaResults(results);
      // Load AI summaries for the first few items
      if (!demoMode) {
        loadAISummariesForItems(results);
      }
    } catch {}
    finally { setLoadingIndia(false); }
  };

  // initial and reactive loads
  useEffect(() => {
    loadMovies(movieTab).catch(() => {});
  }, [movieTab]);
  useEffect(() => {
    loadTV(tvTab).catch(() => {});
  }, [tvTab]);
  useEffect(() => {
    loadAnime(animeTab).catch(() => {});
  }, [animeTab]);
  useEffect(() => {
    loadIndia(indiaTab).catch(() => {});
  }, [indiaTab]);
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${apiBase}/voice/health`);
        if (r.ok) {
          const j = await r.json();
          setVoiceApi(!!(j && j.tts));
        }
      } catch {}
    })();
  }, []);

  // Check for trending summaries on app load
  useEffect(() => {
    if (demoMode) return;

    (async () => {
      try {
        // Check if we need to generate summaries for trending titles
        const r = await fetch(`${apiBase}/trending/summaries/check`, {
          headers: authHeaders()
        });
        if (r.ok) {
          const j = await r.json();
          if (j.needs_generation && j.missing_count > 5) {
            // If many summaries are missing, trigger bulk generation
            fetch(`${apiBase}/trending/summaries/generate`, {
              method: 'POST',
              headers: authHeaders()
            }).catch(() => {}); // Fire and forget
          }
        }
      } catch {}
    })();
  }, [token]);

  const speak = useMemo(
    () =>
      async (text: string) => {
        if (!voiceEnabled || !text) return;
        if (voiceApi) {
          try {
            const resp = await fetch(`${apiBase}/voice/tts`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text }),
            });
            if (resp.ok) {
              const blob = await resp.blob();
              const url = URL.createObjectURL(blob);
              const audio = new Audio(url);
              audio.play().catch(() => {});
              return;
            }
          } catch {}
        }
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          const u = new SpeechSynthesisUtterance(text);
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        }
      },
    [voiceEnabled, voiceApi]
  );

  // Home page no longer hosts the chat input; send is not used here

  const fetchLLMSummary = async (item: any) => {
    try {
      const external_id = item.id?.toString() || '';
      const title = item.title || item.name || '';
      const year = item.release_date || item.first_air_date ?
        new Date(item.release_date || item.first_air_date).getFullYear().toString() : '';
      const media_type = item.name ? 'tv' : 'movie';

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
        return j.summary as string;
      }
    } catch {}
    return 'Could not retrieve AI summary at this time.';
  };

  // Function to check if a summary contains error messages that shouldn't be displayed on cards
  const isErrorMessage = (summary: string): boolean => {
    if (!summary) return false;
    
    const errorPatterns = [
      'error',
      'Error',
      'ERROR',
      'failed',
      'Failed',
      'FAILED',
      'unavailable',
      'not available',
      'not initialized',
      'check your',
      'Check your',
      'API key',
      'api key',
      'generating recommendations',
      'processing your request',
      'encountered an issue',
      'something went wrong',
      'try again',
      'currently unavailable'
    ];
    
    return errorPatterns.some(pattern => 
      summary.toLowerCase().includes(pattern.toLowerCase())
    );
  };

  const fetchTrendingSummary = async (externalId: string): Promise<string | null> => {
    try {
      const r = await fetch(`${apiBase}/trending/summaries/${externalId}`, {
        headers: authHeaders()
      });
      if (r.ok) {
        const j = await r.json();
        // Filter out error messages
        if (j.summary && !isErrorMessage(j.summary)) {
          return j.summary;
        }
      }
    } catch {}
    return null;
  };

  const loadAISummaryForItem = async (item: any) => {
    const externalId = item.id?.toString();
    if (!externalId || loadingSummaries.has(externalId) || aiSummaries[externalId]) {
      return;
    }

    setLoadingSummaries(prev => new Set([...prev, externalId]));

    try {
      let summary = await fetchTrendingSummary(externalId);

      // If no cached summary, generate one
      if (!summary) {
        const title = item.title || item.name || '';
        const year = item.release_date || item.first_air_date ?
          new Date(item.release_date || item.first_air_date).getFullYear().toString() : '';
        const media_type = item.name ? 'tv' : 'movie';

        try {
          const r = await fetch(`${apiBase}/summary/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...authHeaders() },
            body: JSON.stringify({
              external_id: externalId,
              title,
              year,
              media_type,
              source: 'tmdb'
            }),
          });

          if (r.ok) {
            const j = await r.json();
            summary = j.summary;
          }
        } catch {}
      }

      if (summary && !isErrorMessage(summary)) {
        setAiSummaries(prev => ({...prev, [externalId]: summary}));
      }
    } catch {}
    finally {
      setLoadingSummaries(prev => {
        const newSet = new Set(prev);
        newSet.delete(externalId);
        return newSet;
      });
    }
  };

  const loadDetailSummaryForItem = async (item: any) => {
    const externalId = item.id?.toString();
    if (!externalId) return;

    // Skip if already cached (from localStorage or previous fetch)
    if (detailSummaries[externalId]) {
      return;
    }

    try {
      const summary = await fetchLLMSummary(item);
      if (summary && !isErrorMessage(summary)) {
        setDetailSummaries(prev => ({...prev, [externalId]: summary}));
      }
    } catch {}
  };

  const loadAISummariesForItems = async (items: any[]) => {
    // Prefetch detail summaries for ALL items in background
    items.forEach((item, index) => {
      // Stagger requests to avoid overwhelming the backend
      setTimeout(() => loadDetailSummaryForItem(item), index * 500);
    });
  };

  const searchTMDBHome = useCallback(async (query: string) => {
    const q = (query || '').trim();
    if (!q || q.length < 2) {
      setSearchResults([]);
      setSearchMsg('');
      return;
    }

    // Update last search to prevent duplicate debounced search
    lastSearchRef.current = q;

    setSearchMsg('Searching TMDB...');
    setSearchLoading(true);
    try {
      const [movie, tv] = await Promise.all([
        tmdbRequest('/search/movie', { query: q }),
        tmdbRequest('/search/tv', { query: q }),
      ]);
      const all = [
        ...(movie.results || []).map((x:any)=>({ ...x, type:'movie'})),
        ...(tv.results || []).map((x:any)=>({ ...x, type:'tv'})),
      ].sort((a:any,b:any)=> (b.popularity||0)-(a.popularity||0));

      setSearchResults(all.slice(0, 10)); // Show top 10 results

      if (all.length > 0) {
        setSearchMsg(`Found ${all.length} results`);
      } else {
        setSearchMsg('No results found.');
      }
    } catch {
      setSearchMsg('Search failed.');
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  }, []);

  const openDetails = async (item: any) => {
    setDetailsItem(item);
    setDetailsOpen(true);

    const externalId = item.id?.toString();
    // Check if we have a cached summary (from localStorage or previous fetch)
    if (externalId && detailSummaries[externalId]) {
      setDetailsLLM(detailSummaries[externalId]);
      console.log(`✅ Loaded from cache: ${item.title || item.name}`);
    } else {
      // Show loading message
      setDetailsLLM('Loading AI summary...');
      console.log(`⏳ Fetching new summary: ${item.title || item.name}`);
      // Fetch summary if not cached
      const llm = await fetchLLMSummary(item);
      setDetailsLLM(llm);
      // Cache it for next time (will be saved to localStorage automatically)
      if (externalId && llm && !isErrorMessage(llm)) {
        setDetailSummaries(prev => ({...prev, [externalId]: llm}));
      }
    }
  };

  // Ask via LLM is not triggered from home now

  const TrendingGrid = ({ items, kind }: { items: any[]; kind: 'movie' | 'tv' | 'anime' | 'india' }) => (
    <div className="grid-5 card-grid">
      {items.map((it) => {
        const externalId = it.id?.toString();
        const hasDetailCache = externalId && detailSummaries[externalId];

        return (
          <div
            key={it.id}
            onClick={() => openDetails(it)}
            className="item"
            style={{ position: 'relative', cursor: 'pointer' }}
            title={hasDetailCache ? 'Details ready - click to view instantly!' : 'Click to load details'}
          >
            {hasDetailCache && (
              <div
                style={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  background: 'rgba(34, 197, 94, 0.9)',
                  color: 'white',
                  borderRadius: '50%',
                  width: 24,
                  height: 24,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 12,
                  fontWeight: 'bold',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                  zIndex: 10
                }}
                title="Details cached - instant load!"
              >
                ⚡
              </div>
            )}
            <img
              src={it.poster_path ? `${tmdbImageBaseUrl}${it.poster_path}` : 'https://via.placeholder.com/500x750?text=No+Image'}
              alt={it.title || it.name}
              className="responsive"
            />
            <div style={{ padding: 8 }}>
              <div className="title">{it.title || it.name}</div>
              <div style={{ color: '#e6dcc6', opacity: 0.8, fontSize: 11 }}>
                {(it.release_date || it.first_air_date || 'N/A').slice(0, 4)}{' '}
                •{' '}
                {it.genre_ids && it.genre_ids.length > 0 ? genres[String(it.genre_ids[0])] || (it.name ? 'TV' : 'Movie') : it.name ? 'TV' : 'Movie'}
              </div>
              {it.vote_average ? (
                <div style={{ color: '#e6dcc6', fontSize: 12, marginTop: 4 }}>★ {(it.vote_average || 0).toFixed(1)}</div>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div>
      <div className="container">
        <header className="site-header">
          <h1 className="brand">🎬 Cinemizer</h1>
          <p>Your Ultimate AI-Powered Movie & TV Show Companion</p>
        </header>

        <>
          <div className="card" style={{ marginBottom: 12 }}>
            <h3 className="section-title">About Cinemizer Chatbot</h3>
            <p style={{ opacity: 0.85 }}>Ask about movies or shows and get concise overviews, cast/creators, ratings, and similar recommendations.</p>
            <button onClick={() => router.push('/chat')} className="btn btn-primary" style={{ marginTop:8 }}>Open Chatbot</button>
          </div>

          <main style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Quick Search */}
          <section className="card">
            <div style={{ display:'flex', gap:8, alignItems:'center', flexWrap:'wrap' }}>
              <div style={{ flex:1, minWidth:260, position:'relative' }}>
                <input
                  value={searchQuery}
                  onChange={(e)=>setSearchQuery(e.target.value)}
                  onKeyDown={(e)=>{ if(e.key==='Enter') searchTMDBHome(searchQuery); }}
                  placeholder="Search movies & TV shows... (auto-search after typing)"
                  style={{ width:'100%', padding:'10px 40px 10px 14px', borderRadius:8, border:'1px solid rgba(230,220,198,0.2)', background:'rgba(230,220,198,0.05)', color:'#e6dcc6' }}
                />
                {searchQuery.trim() && (
                  <button
                    onClick={() => { setSearchQuery(''); setSearchResults([]); setSearchMsg(''); }}
                    style={{
                      position:'absolute',
                      right:12,
                      top:'50%',
                      transform:'translateY(-50%)',
                      background:'transparent',
                      border:'none',
                      color:'#f4a261',
                      cursor:'pointer',
                      fontSize:18,
                      padding:4,
                      lineHeight:1,
                      opacity:0.7
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                    onMouseLeave={(e) => e.currentTarget.style.opacity = '0.7'}
                  >
                    ✕
                  </button>
                )}
                {searchQuery.trim() && searchQuery !== debouncedSearchQuery && !searchLoading && (
                  <span style={{ position:'absolute', right:40, top:'50%', transform:'translateY(-50%)', fontSize:11, opacity:0.5, color:'#f4a261' }}>
                    ⏳
                  </span>
                )}
              </div>
              <button className="btn btn-primary" onClick={()=>searchTMDBHome(searchQuery)} disabled={searchLoading}>🔍 {searchLoading?'Searching...':'Search'}</button>
            </div>
            {searchMsg && <div className="muted" style={{ marginTop:8 }}>{searchMsg}</div>}

            {/* Search Results List */}
            {searchResults.length > 0 && (
              <div style={{ marginTop: 16, display: 'grid', gap: 12 }}>
                {searchResults.map((item) => {
                  const title = item.title || item.name || 'Unknown';
                  const year = (item.release_date || item.first_air_date || '').split('-')[0];
                  const mediaType = item.type || 'movie';
                  const rating = item.vote_average ? item.vote_average.toFixed(1) : 'N/A';

                  return (
                    <div
                      key={item.id}
                      onClick={() => openDetails(item)}
                      style={{
                        display: 'flex',
                        gap: 12,
                        padding: 12,
                        background: 'rgba(230,220,198,0.03)',
                        borderRadius: 8,
                        border: '1px solid rgba(230,220,198,0.1)',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(230,220,198,0.08)';
                        e.currentTarget.style.borderColor = 'rgba(244,162,97,0.3)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'rgba(230,220,198,0.03)';
                        e.currentTarget.style.borderColor = 'rgba(230,220,198,0.1)';
                      }}
                    >
                      {item.poster_path && (
                        <img
                          src={`https://image.tmdb.org/t/p/w92${item.poster_path}`}
                          alt={title}
                          style={{ width: 60, height: 90, borderRadius: 6, objectFit: 'cover' }}
                        />
                      )}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                          <h4 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{title}</h4>
                          <span style={{ fontSize: 12, opacity: 0.6 }}>
                            {mediaType === 'tv' ? '📺 TV' : '🎬 Movie'}
                          </span>
                        </div>
                        {year && <div style={{ fontSize: 13, opacity: 0.7, marginBottom: 4 }}>{year}</div>}
                        <div style={{ display: 'flex', gap: 12, fontSize: 13 }}>
                          <span>⭐ {rating}</span>
                          {item.popularity && (
                            <span style={{ opacity: 0.6 }}>👥 {Math.round(item.popularity)}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Example searches - only show when no results */}
            {searchResults.length === 0 && (
              <div style={{ display:'flex', gap:8, marginTop:10, flexWrap:'wrap' }}>
                {['Dune: Part Two','The Bear','Oppenheimer','Wednesday','Avatar: The Way of Water','House of the Dragon'].map(ex=> (
                  <button key={ex} className="tab" onClick={()=>{ setSearchQuery(ex); searchTMDBHome(ex); }}>{ex}</button>
                ))}
              </div>
            )}
          </section>
          {/* Trending Movies */}
          <section className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <h3 className="section-title">🎬 Trending Movies</h3>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {[
                  { k: 'trending_movies_day', l: 'Today' },
                  { k: 'trending_movies_week', l: 'This Week' },
                  { k: 'popular_movies', l: 'Popular' },
                  { k: 'top_rated_movies', l: 'Top Rated' },
                  { k: 'now_playing', l: 'Now Playing' },
                  { k: 'upcoming_movies', l: 'Upcoming' },
                ].map((b: any) => (
                  <button
                    key={b.k}
                    onClick={() => setMovieTab(b.k)}
                    className={`tab ${movieTab === b.k ? 'active' : ''}`}
                  >
                    {b.l}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              {loadingMovies ? (
                <div className="loading-grid">
                  {Array.from({length:5}).map((_,i)=> (
                    <div key={i} className="loading-card"><div className="loading-shimmer"/></div>
                  ))}
                </div>
              ) : (
                <TrendingGrid items={movieResults} kind="movie" />
              )}
            </div>
          </section>

          {/* Trending TV */}
          <section className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <h3 className="section-title">📺 Trending TV Shows</h3>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {[
                  { k: 'trending_tv_day', l: 'Today' },
                  { k: 'trending_tv_week', l: 'This Week' },
                  { k: 'popular_tv', l: 'Popular' },
                  { k: 'top_rated_tv', l: 'Top Rated' },
                  { k: 'airing_today', l: 'Airing Today' },
                  { k: 'on_the_air', l: 'On Air' },
                ].map((b: any) => (
                  <button
                    key={b.k}
                    onClick={() => setTvTab(b.k)}
                    className={`tab ${tvTab === b.k ? 'active' : ''}`}
                  >
                    {b.l}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              {loadingTV ? (
                <div className="loading-grid">
                  {Array.from({length:5}).map((_,i)=> (
                    <div key={i} className="loading-card"><div className="loading-shimmer"/></div>
                  ))}
                </div>
              ) : (
                <TrendingGrid items={tvResults} kind="tv" />
              )}
            </div>
          </section>

          {/* Indian Cinema */}
          <section className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <h3 className="section-title">🇮🇳 Indian Cinema</h3>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {[
                  { k: 'bollywood', l: 'Bollywood' },
                  { k: 'south_indian', l: 'South Indian' },
                  { k: 'regional', l: 'Regional' },
                ].map((b: any) => (
                  <button
                    key={b.k}
                    onClick={() => setIndiaTab(b.k)}
                    className={`tab ${indiaTab === b.k ? 'active' : ''}`}
                  >
                    {b.l}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              {loadingIndia ? (
                <div className="loading-grid">
                  {Array.from({length:5}).map((_,i)=> (
                    <div key={i} className="loading-card"><div className="loading-shimmer"/></div>
                  ))}
                </div>
              ) : (
                <TrendingGrid items={indiaResults} kind="india" />
              )}
            </div>
          </section>

          {/* Trending Anime */}
          <section className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <h3 className="section-title">🍥 Trending Anime</h3>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {[
                  { k: 'anime_top_rated', l: 'Top Rated' },
                  { k: 'anime_airing', l: 'Currently Airing' },
                ].map((b: any) => (
                  <button
                    key={b.k}
                    onClick={() => setAnimeTab(b.k)}
                    className={`tab ${animeTab === b.k ? 'active' : ''}`}
                  >
                    {b.l}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              {loadingAnime ? (
                <div className="loading-grid" style={{gridTemplateColumns:'repeat(2,1fr)'}}>
                  {Array.from({length:6}).map((_,i)=> (
                    <div key={i} className="loading-card" style={{height:240}}><div className="loading-shimmer"/></div>
                  ))}
                </div>
              ) : (
                <TrendingGrid items={animeResults} kind="anime" />
              )}
            </div>
          </section>
        </main>
        </>
        {/* Details Modal */}
        {detailsOpen && detailsItem && (
          <div onClick={() => setDetailsOpen(false)} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 9999, display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: 40 }}>
            <div onClick={(e) => e.stopPropagation()} style={{ width: '90%', maxWidth: 900, background: 'rgba(20,15,10,0.98)', border: '1px solid rgba(230,220,198,0.15)', borderRadius: 14, padding: 16, color: '#e6dcc6' }}>
              <div style={{ textAlign: 'right' }}>
                <button onClick={() => setDetailsOpen(false)} style={{ background: 'transparent', color: '#e6dcc6', border: 0, fontSize: 18, cursor: 'pointer' }}>✖</button>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 16 }}>
                <img src={detailsItem.poster_path ? `${tmdbImageBaseUrl}${detailsItem.poster_path}` : 'https://via.placeholder.com/180x270?text=No+Image'} alt="Poster" style={{ width: 180, height: 270, objectFit: 'cover', borderRadius: 8, border: '1px solid rgba(230,220,198,0.25)' }} />
                <div>
                  <h3 style={{ color: '#e6dcc6', margin: 0 }}>{detailsItem.title || detailsItem.name}</h3>
                  <p>Year: {(detailsItem.release_date || detailsItem.first_air_date || 'N/A').slice(0, 4)}</p>
                  <p>Genres: {detailsItem.genre_ids && detailsItem.genre_ids.length > 0 ? genres[String(detailsItem.genre_ids[0])] || (detailsItem.name ? 'TV' : 'Movie') : (detailsItem.name ? 'TV' : 'Movie')}</p>
                  <p>Overview: {detailsItem.overview || 'N/A'}</p>
                  <p>Rating: {detailsItem.vote_average ? detailsItem.vote_average.toFixed(1) : 'N/A'}</p>
                  <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid rgba(230,220,198,0.2)' }}>
                    <ReactMarkdown
                      components={{
                        p: ({children}) => <p style={{ margin: '8px 0', lineHeight: '1.6' }}>{children}</p>,
                        strong: ({children}) => <strong style={{ color: '#f4a261', fontWeight: 600 }}>{children}</strong>,
                        em: ({children}) => <em style={{ fontStyle: 'italic', opacity: 0.9 }}>{children}</em>,
                        ul: ({children}) => <ul style={{ marginLeft: 20, marginTop: 8 }}>{children}</ul>,
                        li: ({children}) => <li style={{ marginBottom: 4 }}>{children}</li>,
                      }}
                    >
                      {detailsLLM}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

