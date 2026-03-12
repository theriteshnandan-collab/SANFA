'use client';

import { useEffect, useState, useMemo } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Upload, LogOut, Grid, Image as ImageIcon, Zap, CheckCircle, Clock } from 'lucide-react';

export default function Dashboard() {
  const supabase = useMemo(() => createClient(), []);
  const [user, setUser] = useState<any>(null);
  const [profile, setProfile] = useState<any>(null);
  const [assets, setAssets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        window.location.href = '/login';
        return;
      }
      setUser(user);

      // Fetch Profile & Assets in parallel
      const [profileRes, assetsRes] = await Promise.all([
        supabase.from('profiles').select('*').eq('id', user.id).single(),
        supabase.from('protected_images').select('*').eq('user_id', user.id).order('created_at', { ascending: false })
      ]);

      setProfile(profileRes.data);
      setAssets(assetsRes.data || []);
      setLoading(false);
    };

    fetchData();

    // REAL-TIME SUBSCRIPTION: Listen for job status changes
    const channel = supabase
      .channel('schema-db-changes')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'protected_images', filter: `user_id=eq.${user?.id}` }, 
        (payload) => {
          setAssets(prev => {
            const index = prev.findIndex(a => a.id === (payload.new as any).id);
            if (index > -1) {
              const next = [...prev];
              next[index] = payload.new;
              return next;
            }
            return [payload.new, ...prev];
          });
        }
      )
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [user?.id]);

  const handleProtect = async () => {
    setIsUploading(true);
    try {
      // 1. SIMULATE UPLOAD (In prod: Upload to Supabase Storage first)
      const mockUrl = `https://picsum.photos/seed/${Math.random()}/800/600`;
      
      // 2. TRIGGER API
      const res = await fetch('/api/protect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageUrl: mockUrl,
          fileName: `Artwork_${Date.now()}.png`
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

    } catch (err) {
      console.error('Handoff Failed:', err);
      alert('System handoff failed. Check logs.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    window.location.href = '/login';
  };

  if (loading) return (
    <div className="min-h-screen bg-[#0F0F0F] flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <Zap className="w-8 h-8 text-[#C9A84C] animate-pulse" />
        <span className="text-white/30 text-sm font-sans tracking-widest uppercase">Initializing Enclave...</span>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0F0F0F] text-white">
      {/* Sidebar / Nav */}
      <nav className="border-b border-white/5 bg-[#141414] px-8 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#C9A84C]/20 rounded-xl flex items-center justify-center border border-[#C9A84C]/30">
            <Shield className="text-[#C9A84C] w-6 h-6" />
          </div>
          <span className="text-xl font-serif tracking-tight">SANFA <span className="text-[#C9A84C]">CLOUD</span></span>
        </div>

        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 px-4 py-2 bg-black/40 rounded-full border border-white/5">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs font-sans text-white/50 uppercase tracking-widest">{user?.email}</span>
          </div>
          <button 
            onClick={handleLogout}
            className="text-white/30 hover:text-white transition-all flex items-center gap-2 group"
          >
            <LogOut className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-8 py-12">
        {/* Header Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {[
            { label: 'Credits Remaining', val: profile?.credits_remaining || '0', color: '#C9A84C' },
            { label: 'Protected Assets', val: assets.length, color: '#10b981' },
            { label: 'Security Tier', val: profile?.subscription_status?.toUpperCase() || 'FREE', color: '#fff' }
          ].map((stat, i) => (
            <div key={i} className="bg-[#1A1A1A] border border-white/5 p-8 rounded-2xl relative overflow-hidden group hover:border-[#C9A84C]/30 transition-all">
              <div className="absolute top-0 right-0 w-32 h-32 bg-[#C9A84C]/5 rounded-bl-[100px] -mr-8 -mt-8 group-hover:bg-[#C9A84C]/10 transition-all" />
              <p className="text-white/40 text-xs font-sans uppercase tracking-[0.2em] mb-4">{stat.label}</p>
              <h3 className="text-4xl font-serif" style={{ color: stat.color }}>{stat.val}</h3>
            </div>
          ))}
        </div>

        {/* Action Center */}
        <div className="bg-[#1A1A1A] border border-white/5 rounded-3xl p-12 text-center relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-[#C9A84C]/5 via-transparent to-transparent opacity-50" />
          <div className="relative z-10">
            <div className={`w-20 h-20 bg-[#C9A84C]/10 rounded-3xl flex items-center justify-center mx-auto mb-8 border border-[#C9A84C]/20 transition-transform duration-500 ${isUploading ? 'animate-spin' : 'group-hover:scale-110'}`}>
              <Upload className="text-[#C9A84C] w-8 h-8" />
            </div>
            <h2 className="text-3xl font-serif mb-4">Protect New Masterpiece</h2>
            <p className="text-white/40 max-w-lg mx-auto mb-10 text-lg leading-relaxed">
              Upload your high-resolution artwork to inject Engine V5 Anti-AI poisoning. 
              Decoupled GPU workers will process your request in the background.
            </p>
            <button 
              onClick={handleProtect}
              disabled={isUploading}
              className="bg-white text-black font-semibold px-12 py-4 rounded-full hover:bg-[#C9A84C] hover:text-black disabled:opacity-50 transition-all flex items-center gap-3 mx-auto shadow-xl hover:shadow-[#C9A84C]/20"
            >
              {isUploading ? 'Engaging Engine...' : 'Select & Protect'}
              <Zap className="w-4 h-4 fill-current" />
            </button>
          </div>
        </div>

        {/* Asset Grid */}
        <div className="mt-20">
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-2xl font-serif flex items-center gap-4">
              <Grid className="text-[#C9A84C] w-6 h-6" />
              Vault Library
            </h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {assets.map((asset) => (
              <div key={asset.id} className="bg-[#1A1A1A] border border-white/5 rounded-2xl p-4 group hover:border-[#C9A84C]/30 transition-all">
                <div className="aspect-square bg-black/40 rounded-xl mb-4 overflow-hidden relative">
                  {asset.status === 'completed' ? (
                    <img src={asset.protected_url || asset.original_url} alt={asset.original_name} className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity" />
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center gap-3">
                      <Clock className="w-8 h-8 text-white/10 animate-spin" />
                      <span className="text-[10px] uppercase tracking-widest text-white/20">Processing</span>
                    </div>
                  )}
                  <div className="absolute top-2 right-2">
                    {asset.status === 'completed' ? <CheckCircle className="w-4 h-4 text-[#10b981]" /> : <Zap className="w-4 h-4 text-[#C9A84C] animate-pulse" />}
                  </div>
                </div>
                <h4 className="text-sm font-sans font-medium truncate mb-1">{asset.original_name}</h4>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-white/30 uppercase tracking-tighter">
                    {new Date(asset.created_at).toLocaleDateString()}
                  </span>
                  <span className="text-[10px] text-[#C9A84C] font-bold">
                    {asset.clip_distance ? `${Math.round(asset.clip_distance * 100)}% Confusion` : 'PROTECTING'}
                  </span>
                </div>
              </div>
            ))}

            {assets.length === 0 && !isUploading && (
              <div className="col-span-full py-20 bg-white/[0.02] border border-dashed border-white/5 rounded-3xl flex flex-col items-center justify-center gap-4">
                <ImageIcon className="w-12 h-12 text-white/5" />
                <span className="text-white/20 font-sans tracking-widest uppercase text-xs">No assets in enclave</span>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
