'use client';

import { useState } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Mail, Lock, ArrowRight, Sparkles, Fingerprint } from 'lucide-react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';

export const dynamic = 'force-dynamic';

// Apple-Standard Spring Configurations
const organicSpring = { type: "spring", stiffness: 300, damping: 30, mass: 1 };
const entranceSpring = { type: "spring", stiffness: 100, damping: 20, mass: 1 };

const fadeInUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: organicSpring
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
    <div className="min-h-screen bg-[#030304] flex items-center justify-center p-8 selection:bg-gold-500/40 text-white overflow-hidden relative">
      {/* God-Tier Ambience */}
      <div className="fixed inset-0 pointer-events-none noise-fine z-50" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[600px] bg-gold-500/[0.03] blur-[140px] rounded-full z-0" />

      <motion.div 
        initial={{ opacity: 0, scale: 0.98, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={entranceSpring}
        className="w-full max-w-xl relative z-10"
      >
        <div className="glass-card-v2 rounded-[64px] p-20 shadow-[0_0_100px_rgba(0,0,0,0.5)] border-white/[0.03] relative overflow-hidden text-center">
          <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-gold-500/30 to-transparent" />
          
          <div className="flex flex-col items-center mb-16">
            <motion.div 
              initial={{ rotate: -15, scale: 0.8 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={organicSpring}
              className="w-24 h-24 glass-v2 rounded-[3rem] flex items-center justify-center mb-8 border border-white/10"
            >
              <Shield className="text-gold-500 w-12 h-12" />
            </motion.div>
            
            <motion.h1 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              className="text-4xl font-serif text-aura mb-4 tracking-tight"
            >
              Protocol Access
            </motion.h1>
            
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="flex items-center gap-4"
            >
              <Fingerprint className="w-4 h-4 text-gold-500/30" />
              <p className="text-[11px] font-black uppercase tracking-[0.6em] text-white/20">Secure Asset Enclave Entry</p>
            </motion.div>
          </div>

          <form className="space-y-12 text-left">
            <motion.div variants={fadeInUp} initial="initial" animate="animate" transition={{ delay: 0.2 }}>
              <label className="block text-[11px] font-black uppercase tracking-[0.5em] text-white/30 mb-4 ml-2">Email Identity</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-6 flex items-center pointer-events-none text-white/20 group-focus-within:text-gold-500 transition-colors">
                  <Mail className="w-5 h-5" />
                </div>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-white/[0.02] border border-white/5 rounded-3xl py-6 pl-16 pr-8 text-white placeholder:text-white/10 focus:outline-none focus:border-gold-500/30 focus:bg-white/[0.04] transition-all font-sans text-base"
                  placeholder="artist@sanfa.id"
                  required
                />
              </div>
            </motion.div>

            <motion.div variants={fadeInUp} initial="initial" animate="animate" transition={{ delay: 0.3 }}>
              <label className="block text-[11px] font-black uppercase tracking-[0.5em] text-white/30 mb-4 ml-2">Security Hash</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-6 flex items-center pointer-events-none text-white/20 group-focus-within:text-gold-500 transition-colors">
                  <Lock className="w-5 h-5" />
                </div>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-white/[0.02] border border-white/5 rounded-3xl py-6 pl-16 pr-8 text-white placeholder:text-white/10 focus:outline-none focus:border-gold-500/30 focus:bg-white/[0.04] transition-all font-sans text-base"
                  placeholder="••••••••"
                  required
                />
              </div>
            </motion.div>

            <AnimatePresence mode="wait">
              {message && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={`p-6 rounded-3xl text-[11px] font-black uppercase tracking-[0.4em] ${message.type === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}
                >
                  <div className="flex items-center gap-4">
                    <Sparkles className="w-5 h-5" />
                    {message.text}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div 
              variants={fadeInUp} 
              initial="initial" 
              animate="animate" 
              transition={{ delay: 0.4 }}
              className="flex gap-6 pt-6"
            >
              <button
                onClick={handleLogin}
                disabled={loading}
                className="flex-[2] bg-white hover:bg-gold-400 text-black font-black py-6 rounded-3xl transition-all flex items-center justify-center gap-4 group disabled:opacity-30 shadow-2xl uppercase tracking-widest"
              >
                Engage
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
              <button
                onClick={handleSignUp}
                disabled={loading}
                className="flex-1 bg-white/[0.03] hover:bg-white/5 text-white font-black py-6 rounded-3xl border border-white/10 transition-all disabled:opacity-30 text-xs uppercase tracking-[0.3em]"
              >
                Join
              </button>
            </motion.div>
          </form>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
            className="mt-20 pt-10 border-t border-white/[0.03]"
          >
            <Link href="/" className="text-white/20 hover:text-gold-500 transition-all text-xs font-black uppercase tracking-[0.8em]">
              Core Protocol
            </Link>
          </motion.div>
        </div>
        
        {/* Verification Status */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-12 flex justify-center"
        >
          <div className="flex items-center gap-4 px-6 py-3 bg-white/[0.03] rounded-full border border-white/5">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping" />
            <span className="text-[10px] font-black text-white/20 uppercase tracking-[0.5em]">Identity Enclave Protection Active</span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
