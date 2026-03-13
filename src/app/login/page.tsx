'use client';

import { useState } from 'react';
import { createClient } from '@/lib/supabase-browser';
import { Shield, Mail, Lock, ArrowRight, Sparkles, Fingerprint } from 'lucide-react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';

const lemonSpring = { type: "spring", stiffness: 400, damping: 30, mass: 1 } as any;

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: lemonSpring
} as any;

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
      setMessage({ type: 'success', text: 'Identity verified. Accessing vault...' });
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
      setMessage({ type: 'success', text: 'Check your email for the verification link.' });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA] flex items-center justify-center p-6 selection:bg-lemon/10 text-[#111111]">
      
      <motion.div 
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={lemonSpring}
        className="w-full max-w-md relative z-10"
      >
        <div className="lemon-card p-12 md:p-16 text-center">
          
          <div className="flex flex-col items-center mb-12">
            <motion.div 
              initial={{ rotate: -10, scale: 0.8 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={lemonSpring}
              className="w-16 h-16 bg-lemon rounded-2xl flex items-center justify-center mb-8 shadow-lg shadow-lemon/20"
            >
              <Shield className="text-white w-8 h-8" />
            </motion.div>
            
            <motion.h1 
              variants={fadeInUp}
              initial="initial"
              animate="animate"
              className="text-3xl font-black mb-4 tracking-tighter"
            >
              Welcome back.
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="text-gray-400 font-bold text-xs uppercase tracking-widest"
            >
              Enter the SANFA Enclave
            </motion.p>
          </div>

          <form className="space-y-8 text-left">
            <motion.div variants={fadeInUp} initial="initial" animate="animate" transition={{ delay: 0.2 }}>
              <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-gray-400 mb-2 ml-1">Email</label>
              <div className="relative group">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-100 rounded-2xl py-4 px-6 text-[#111111] placeholder:text-gray-300 focus:outline-none focus:border-lemon/30 focus:bg-white transition-all font-sans text-sm"
                  placeholder="artist@sanfa.id"
                  required
                />
              </div>
            </motion.div>

            <motion.div variants={fadeInUp} initial="initial" animate="animate" transition={{ delay: 0.3 }}>
              <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-gray-400 mb-2 ml-1">Password</label>
              <div className="relative group">
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-100 rounded-2xl py-4 px-6 text-[#111111] placeholder:text-gray-300 focus:outline-none focus:border-lemon/30 focus:bg-white transition-all font-sans text-sm"
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
                  className={`p-4 rounded-xl text-[10px] font-black uppercase tracking-widest ${message.type === 'success' ? 'bg-green-50 text-green-500 border border-green-100' : 'bg-red-50 text-red-500 border border-red-100'}`}
                >
                  {message.text}
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div 
              variants={fadeInUp} 
              initial="initial" 
              animate="animate" 
              transition={{ delay: 0.4 }}
              className="flex flex-col gap-4 pt-4"
            >
              <button
                onClick={handleLogin}
                disabled={loading}
                className="btn-lemon w-full py-5 rounded-2xl flex items-center justify-center gap-4 group disabled:opacity-50"
              >
                Sign In
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
              <button
                onClick={handleSignUp}
                disabled={loading}
                className="w-full bg-white text-gray-400 font-black py-4 rounded-2xl border border-gray-100 hover:bg-gray-50 transition-all disabled:opacity-50 text-xs uppercase tracking-widest"
              >
                Create Account
              </button>
            </motion.div>
          </form>

          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-12 pt-8 border-t border-gray-50"
          >
            <Link href="/" className="text-gray-300 hover:text-lemon transition-colors text-[10px] font-bold uppercase tracking-widest">
              Back to Home
            </Link>
          </motion.div>
        </div>
        
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-8 flex justify-center"
        >
          <div className="flex items-center gap-3 px-4 py-2 bg-white rounded-full shadow-sm border border-gray-100">
            <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
            <span className="text-[9px] font-black text-gray-400 uppercase tracking-widest">Sovereign Protection Active</span>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}

