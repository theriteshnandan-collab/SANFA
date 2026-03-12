'use client';

import { useState } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Mail, Lock, ArrowRight } from 'lucide-react';
import Link from 'next/link';

export default function LoginPage() {
  const supabase = createClient();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      setMessage({ type: 'error', text: error.message });
    } else {
      setMessage({ type: 'success', text: 'Success! Redirecting...' });
      window.location.href = '/dashboard';
    }
    setLoading(false);
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) {
      setMessage({ type: 'error', text: error.message });
    } else {
      setMessage({ type: 'success', text: 'Check your email for the confirmation link!' });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#0F0F0F] flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-[#1A1A1A] border border-white/10 rounded-2xl p-8 shadow-2xl">
        <div className="flex flex-col items-center mb-8">
          <div className="w-16 h-16 bg-[#C9A84C]/20 rounded-2xl flex items-center justify-center mb-4 border border-[#C9A84C]/30">
            <Shield className="text-[#C9A84C] w-8 h-8" />
          </div>
          <h1 className="text-3xl font-serif text-white text-center">Protocol Access</h1>
          <p className="text-white/50 text-center mt-2">Enter the SANFA Secure Enclave</p>
        </div>

        <form className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">Email Identity</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-[#C9A84C]/50 transition-all font-sans"
                placeholder="artist@sanfa.dev"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-white/70 mb-2">Security Key</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white placeholder:text-white/20 focus:outline-none focus:border-[#C9A84C]/50 transition-all font-sans"
                placeholder="••••••••"
                required
              />
            </div>
          </div>

          {message && (
            <div className={`p-4 rounded-xl text-sm ${message.type === 'success' ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
              {message.text}
            </div>
          )}

          <div className="flex gap-4">
            <button
              onClick={handleLogin}
              disabled={loading}
              className="flex-1 bg-[#C9A84C] hover:bg-[#B8963D] text-black font-semibold py-3 rounded-xl transition-all flex items-center justify-center gap-2 group disabled:opacity-50"
            >
              Log In
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={handleSignUp}
              disabled={loading}
              className="flex-1 bg-white/5 hover:bg-white/10 text-white font-semibold py-3 rounded-xl border border-white/10 transition-all disabled:opacity-50"
            >
              Sign Up
            </button>
          </div>
        </form>

        <div className="mt-8 pt-8 border-t border-white/5 text-center">
          <Link href="/" className="text-white/30 hover:text-white/60 text-sm transition-all">
            Return to Core Protocol
          </Link>
        </div>
      </div>
    </div>
  );
}
