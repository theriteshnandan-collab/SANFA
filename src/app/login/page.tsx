'use client';

import { useState } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Mail, Lock, ArrowRight, Sparkles, Fingerprint } from 'lucide-react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';

export const dynamic = 'force-dynamic';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] }
};

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
      setMessage({ type: 'success', text: 'Identity verified. Decrypting enclave...' });
      setTimeout(() => {
        window.location.href = '/dashboard';
      }, 1000);
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
      setMessage({ type: 'success', text: 'Verification link dispatched to your secure mail.' });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0b] flex items-center justify-center p-6 selection:bg-gold-500/30 selection:text-white overflow-hidden relative">
      {/* Background Ambience */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] noise z-50" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gold-500/5 blur-[160px] rounded-full opacity-30" />

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-md relative z-10"
      >
        <div className="glass-card rounded-[40px] p-10 shadow-2xl shadow-black/50 border-white/5 relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-gold-500/20 to-transparent" />
          
          <div className="flex flex-col items-center mb-12">
            <motion.div 
              initial={{ rotate: -10, scale: 0.9 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={{ delay: 0.2, type: 'spring' }}
              className="w-20 h-20 bg-gold-500/10 rounded-[32px] flex items-center justify-center mb-6 border border-gold-500/20 shadow-xl shadow-gold-500/5"
            >
              <Shield className="text-gold-500 w-10 h-10" />
            </motion.div>
            <motion.h1 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.3 }}
              className="text-3xl font-serif text-white text-center tracking-tight"
            >
              Protocol Access
            </motion.h1>
            <motion.div 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.4 }}
              className="flex items-center gap-2 mt-3"
            >
              <Fingerprint className="w-3.5 h-3.5 text-gold-500/40" />
              <p className="text-[10px] font-bold uppercase tracking-[0.4em] text-white/30">Secure Infrastructure Entry</p>
            </motion.div>
          </div>

          <form className="space-y-8">
            <motion.div 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.5 }}
            >
              <label className="block text-[10px] font-black uppercase tracking-[0.3em] text-white/40 mb-3 ml-1">Email Identity</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none transition-colors group-focus-within:text-gold-500 text-white/20">
                  <Mail className="w-5 h-5" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-white/[0.03] border border-white/5 rounded-2xl py-4 pl-12 pr-6 text-white placeholder:text-white/10 focus:outline-none focus:border-gold-500/50 focus:bg-white/[0.05] transition-all font-sans text-sm"
                  placeholder="artist@sanfa.dev"
                  required
                />
              </div>
            </motion.div>

            <motion.div 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.6 }}
            >
              <label className="block text-[10px] font-black uppercase tracking-[0.3em] text-white/40 mb-3 ml-1">Security Key</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none transition-colors group-focus-within:text-gold-500 text-white/20">
                  <Lock className="w-5 h-5" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-white/[0.03] border border-white/5 rounded-2xl py-4 pl-12 pr-6 text-white placeholder:text-white/10 focus:outline-none focus:border-gold-500/50 focus:bg-white/[0.05] transition-all font-sans text-sm"
                  placeholder="••••••••"
                  required
                />
              </div>
            </motion.div>

            <AnimatePresence mode="wait">
              {message && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className={`p-4 rounded-2xl text-[11px] font-bold uppercase tracking-widest ${message.type === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    {message.text}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              transition={{ delay: 0.7 }}
              className="flex gap-4 pt-4"
            >
              <button
                onClick={handleLogin}
                disabled={loading}
                className="flex-[2] bg-white hover:bg-gold-500 text-black font-bold py-4 rounded-2xl transition-all flex items-center justify-center gap-3 group disabled:opacity-50 shadow-xl hover:shadow-gold-500/20"
              >
                Log In
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
              <button
                onClick={handleSignUp}
                disabled={loading}
                className="flex-1 bg-white/5 hover:bg-white/10 text-white font-bold py-4 rounded-2xl border border-white/10 transition-all disabled:opacity-50 text-xs uppercase tracking-widest"
              >
                Join
              </button>
            </motion.div>
          </form>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="mt-12 pt-8 border-t border-white/5 text-center"
          >
            <Link href="/" className="text-white/20 hover:text-gold-500 transition-all text-[10px] font-black uppercase tracking-[0.3em]">
              Return to Core Protocol
            </Link>
          </motion.div>
        </div>
        
        {/* Verification Tag */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
          className="mt-8 flex justify-center"
        >
          <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full border border-white/5">
            <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
            <span className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em]">AES-256 Auth Active</span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
