'use client';

import { Shield, Sparkles, Zap, Lock, Globe, Activity, Cpu, ArrowRight, ShieldCheck, Fingerprint, Eye, ZapOff } from 'lucide-react';
import Link from 'next/link';
import { motion, useScroll, useTransform, useSpring, AnimatePresence } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';

// Apple-Standard Spring Configurations
const organicSpring = { type: "spring", stiffness: 300, damping: 30, mass: 1 } as any;
const entranceSpring = { type: "spring", stiffness: 100, damping: 20, mass: 1 } as any;


const fadeInUp: any = {
  hidden: { opacity: 0, y: 40 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { ...organicSpring }
  }
};

const staggerContainer: any = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.3
    }
  }
};

export default function Home() {
  const containerRef = useRef(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const smoothProgress = useSpring(scrollYProgress, { stiffness: 100, damping: 30 });
  const opacity = useTransform(smoothProgress, [0, 0.15], [1, 0]);
  const scale = useTransform(smoothProgress, [0, 0.15], [1, 0.92]);

  if (!isMounted) return <div className="min-h-screen bg-[#030304]" />;

  return (
    <div ref={containerRef} className="min-h-screen bg-[#030304] selection:bg-gold-500/40 text-white overflow-x-hidden font-sans">
      {/* God-Tier Atmosphere */}
      <div className="fixed inset-0 pointer-events-none noise-fine z-[100]" />
      <div className="fixed inset-0 pointer-events-none glow-spectral opacity-50 z-0" />
      
      {/* Navigation: Airy & Minimal */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ ...entranceSpring, delay: 0.1 }}
        className="px-12 py-10 flex items-center justify-between fixed top-0 w-full z-[110] pointer-events-none"
      >
        <div className="flex items-center gap-6 pointer-events-auto group">
          <div className="w-12 h-12 glass-v2 rounded-2xl flex items-center justify-center group-hover:scale-105 transition-all duration-700 relative overflow-hidden">
            <Shield className="text-gold-500 w-6 h-6 z-10" />
          </div>
          <div className="flex flex-col">
            <span className="text-2xl font-serif tracking-tight leading-none">SANFA</span>
            <span className="text-[10px] uppercase tracking-[0.6em] text-white/20 font-bold mt-2">Protocol Access</span>
          </div>
        </div>
        
        <div className="flex items-center gap-14 pointer-events-auto">
          <div className="hidden md:flex items-center gap-12">
            {['Architecture', 'Manifesto', 'Security'].map((item) => (
              <Link key={item} href={`#${item.toLowerCase()}`} className="text-[11px] font-bold text-white/20 hover:text-white transition-all uppercase tracking-[0.3em]">{item}</Link>
            ))}
          </div>
          <div className="flex items-center gap-6">
            <Link href="/login" className="glass-v2 px-10 py-4 rounded-full text-xs font-bold uppercase tracking-[0.2em] hover:bg-white/5 transition-all text-white/60 hover:text-white">Access</Link>
            <Link href="/login" className="bg-white text-black px-12 py-4 rounded-full text-xs font-black uppercase tracking-[0.3em] hover:scale-105 active:scale-95 transition-all shadow-2xl">Start Protection</Link>
          </div>
        </div>
      </motion.nav>

      <main className="relative z-10">
        <section className="relative min-h-[110vh] flex flex-col items-center justify-center px-12">
          <motion.div 
            style={{ opacity, scale }}
            variants={staggerContainer as any}
            initial="hidden"
            animate="visible"
            className="max-w-7xl mx-auto text-center"
          >
            <motion.div variants={fadeInUp as any} className="inline-flex items-center gap-4 px-8 py-3 rounded-full bg-white/[0.03] border border-white/[0.05] mb-12">
              <Sparkles className="w-4 h-4 text-gold-500" />
              <span className="text-[11px] font-bold text-white/40 uppercase tracking-[0.5em]">System V6.0 Ascent Active</span>
            </motion.div>

            <motion.h1 
              variants={fadeInUp as any}
              className="text-[var(--font-size-3xl)] font-serif mb-12 tracking-tighter leading-[0.8] text-aura py-4"
            >
              The Invisible <br /> <span className="text-gold-500 italic">Sovereignty.</span>
            </motion.h1>

            <motion.p 
              variants={fadeInUp as any}
              className="text-white/30 max-w-4xl mx-auto text-xl md:text-3xl font-sans leading-relaxed mb-16 tracking-tight text-balance"
            >
              We mathematically scramble every feature kernel to ensure your masterpiece is <span className="text-white/80 italic">mathematically indigestible</span> for neural networks.
            </motion.p>

            <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row items-center justify-center gap-12">
              <Link href="/login" className="group bg-white text-black px-16 py-8 rounded-full text-xl font-black uppercase tracking-widest hover:scale-[1.03] active:scale-95 transition-all flex items-center gap-4 shadow-2xl">
                Engage Protection
                <ArrowRight className="w-6 h-6" />
              </Link>
              <Link href="#manifesto" className="text-[12px] font-black uppercase tracking-[0.5em] text-white/20 hover:text-white transition-all flex items-center gap-4 border-b border-white/5 pb-2">
                Manifesto 2026
              </Link>
            </motion.div>
          </motion.div>
        </section>

        <section id="architecture" className="py-60 relative overflow-hidden">
          <div className="max-w-7xl mx-auto px-12">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-40 items-center">
              <motion.div 
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-200px" }}
                variants={staggerContainer as any}
              >
                <motion.span variants={fadeInUp as any} className="text-gold-500 font-bold text-[11px] uppercase tracking-[0.6em] block mb-8">God-Tier Infrastructure</motion.span>
                <motion.h2 variants={fadeInUp as any} className="text-[var(--font-size-2xl)] font-serif mb-12 leading-[0.9]">Serverless <br /> <span className="text-white/20">Frequency War.</span></motion.h2>
                <motion.p variants={fadeInUp as any} className="text-white/30 text-2xl mb-20 leading-relaxed tracking-tight">
                  By routing every protection job through decentralized spectral enclaves, we eliminate latent recognition signatures before they can be scraped.
                </motion.p>
                
                <div className="space-y-14">
                  {[
                    { icon: Globe, title: 'Edge Supremacy', desc: 'Direct-to-enclave ingestion at the speed of light.' },
                    { icon: Cpu, title: 'H100 Grid', desc: 'Serverless compute clusters processing at the frequency level.' },
                    { icon: Activity, title: 'Spectral Feedback', desc: 'Real-time mathematical distance metrics.' }
                  ].map((item, i) => (
                    <motion.div 
                      key={i}
                      variants={fadeInUp as any}
                      className="flex items-start gap-10 group"
                    >
                      <div className="w-16 h-16 rounded-3xl glass-v2 flex items-center justify-center border border-white/5 group-hover:border-gold-500/50 transition-all duration-700">
                        <item.icon className="text-gold-500 w-7 h-7" />
                      </div>
                      <div>
                        <h4 className="font-serif text-3xl mb-3 text-white/90">{item.title}</h4>
                        <p className="text-white/20 text-lg">{item.desc}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>

              <div className="grid grid-cols-2 gap-8 relative p-4">
                <div className="space-y-8">
                  <motion.div 
                    whileHover={{ y: -8, scale: 1.02 }}
                    transition={organicSpring}
                    className="h-64 glass-card-v2 rounded-[48px] p-12 flex flex-col justify-end"
                  >
                    <span className="text-gold-500 font-serif text-6xl mb-4">99.9<span className="text-2xl opacity-40">%</span></span>
                    <span className="text-white/20 text-[11px] font-bold uppercase tracking-[0.4em]">Uptime</span>
                  </motion.div>
                  <motion.div 
                    whileHover={{ y: -8, scale: 1.02 }}
                    transition={organicSpring}
                    className="h-96 glass-card-v2 rounded-[48px] p-12 bg-gold-500/[0.04] border-gold-500/10 flex flex-col items-center justify-center text-center gap-10"
                  >
                    <Zap className="text-gold-500 w-20 h-20 animate-pulse" />
                    <div className="flex flex-col gap-2">
                       <span className="text-white font-serif text-5xl">200K</span>
                       <span className="text-white/20 text-[11px] font-bold uppercase tracking-[0.4em]">Jobs / Mo</span>
                    </div>
                  </motion.div>
                </div>
                <div className="space-y-8 pt-20">
                  <motion.div 
                    whileHover={{ y: -8, scale: 1.02 }}
                    transition={organicSpring}
                    className="h-96 glass-card-v2 rounded-[48px] p-12 overflow-hidden relative group"
                  >
                    <div className="absolute inset-0 bg-white/[0.02] scale-y-0 group-hover:scale-y-100 transition-transform duration-1000 origin-bottom" />
                    <div className="relative z-20 h-full flex flex-col justify-between">
                      <Fingerprint className="text-white/40 w-16 h-16 group-hover:text-gold-500 transition-colors" />
                      <div className="flex flex-col gap-4">
                        <span className="text-white font-serif text-4xl leading-tight">C2PA Guard</span>
                        <span className="text-white/20 text-[11px] font-bold uppercase tracking-[0.4em]">Immutable Origin</span>
                      </div>
                    </div>
                  </motion.div>
                  <motion.div 
                    whileHover={{ y: -8, scale: 1.02 }}
                    transition={organicSpring}
                    className="h-64 glass-card-v2 rounded-[48px] p-12 flex flex-col justify-center gap-6"
                  >
                    <div className="flex items-center gap-4">
                       <div className="w-3 h-3 bg-gold-500 rounded-full animate-ping" />
                       <span className="text-gold-500 text-[11px] font-bold uppercase tracking-widest">Spectral Active</span>
                    </div>
                    <span className="text-white font-serif text-3xl">6ms Feedback</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="security" className="py-60 px-12 text-center bg-black/20">
          <div className="max-w-4xl mx-auto mb-32">
            <span className="text-gold-500 font-bold text-[11px] uppercase tracking-[0.6em] block mb-10">Defense Protocols</span>
            <h2 className="text-[var(--font-size-2xl)] font-serif text-aura">Impossible Extraction.</h2>
          </div>

          <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-12 text-center px-4">
            {[
              { icon: Eye, title: 'Kernel Void', desc: 'Inverting the statistical kernels used by AI to recognize human-made features.' },
              { icon: Lock, title: 'Spectral Noise', desc: 'Mathematical interference in the frequency domain, invisible to the eye, lethal to the model.' },
              { icon: ZapOff, title: 'Style Decay', desc: 'Ensuring any extraction attempts cause catastrophic model failure and forgetting.' }
            ].map((item, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ ...organicSpring, delay: i * 0.15 }}
                className="glass-card-v2 rounded-[56px] p-20 hover:border-gold-500/20 group"
              >
                <div className="w-24 h-24 glass-v2 rounded-[40px] flex items-center justify-center mb-12 mx-auto group-hover:bg-white/[0.05] transition-all duration-700">
                  <item.icon className="text-white/20 w-12 h-12 group-hover:text-gold-500 transition-colors" />
                </div>
                <h4 className="text-4xl font-serif mb-8 text-white">{item.title}</h4>
                <p className="text-white/20 text-xl leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </section>

        <section id="manifesto" className="py-60 px-12">
           <div className="max-w-6xl mx-auto glass-card-v2 rounded-[80px] p-32 border-white/[0.03] relative overflow-hidden">
               <Shield className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-gold-500/[0.02] w-[800px] h-[800px] pointer-events-none" />
               <div className="relative z-10 text-center">
                  <h2 className="text-[var(--font-size-2xl)] font-serif mb-24 tracking-tighter text-aura">Sovereign <span className="text-gold-500">Manifesto.</span></h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-32 text-left">
                     <div className="space-y-20">
                        <div className="space-y-6">
                           <span className="text-gold-500/40 font-serif text-5xl block italic mb-4">01.</span>
                           <h3 className="text-4xl font-serif">Artificial Plunder.</h3>
                           <p className="text-white/30 text-2xl leading-relaxed">Neural networks treat human genius as un-mined data. We treat it as digital property.</p>
                        </div>
                        <div className="space-y-6">
                           <span className="text-gold-500/40 font-serif text-5xl block italic mb-4">02.</span>
                           <h3 className="text-4xl font-serif">Mathematical Sanction.</h3>
                           <p className="text-white/30 text-2xl leading-relaxed">If the law cannot protect the artist, the math must. SANFA is the technical deterrent.</p>
                        </div>
                     </div>
                     <div className="flex flex-col justify-center items-center p-20 glass-v2 rounded-[64px] border-white/5 gap-12">
                        <p className="text-gold-500 font-serif text-4xl italic text-center leading-[1.1]">
                           "Humanity is not <br /> training data."
                        </p>
                        <Link href="/login" className="bg-white text-black px-16 py-6 rounded-full text-sm font-black uppercase tracking-[0.4em] hover:scale-105 active:scale-95 transition-all">Engage Shield</Link>
                     </div>
                  </div>
               </div>
           </div>
        </section>
      </main>

      <footer className="py-40 border-t border-white/[0.03] bg-black/40 px-12">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-24 text-center md:text-left">
          <div className="flex flex-col gap-6">
            <div className="flex items-center gap-4 justify-center md:justify-start">
               <Shield className="text-gold-500/40 w-8 h-8" />
               <span className="text-3xl font-serif tracking-tight">SANFA</span>
            </div>
            <p className="text-white/20 text-sm max-w-sm font-medium leading-relaxed uppercase tracking-widest">
              Securing the human legacy in the age of algorithmic expansion.
            </p>
          </div>
          
          <div className="grid grid-cols-2 gap-16 md:gap-32">
             <div className="flex flex-col gap-4">
                <span className="text-gold-500 text-6xl font-serif">200K</span>
                <span className="text-white/10 text-[10px] uppercase font-black tracking-[0.6em]">Assets Secured</span>
             </div>
             <div className="flex flex-col gap-4">
                <span className="text-white text-6xl font-serif">99.9%</span>
                <span className="text-white/10 text-[10px] uppercase font-black tracking-[0.6em]">Invisibility Rate</span>
             </div>
          </div>

          <div className="flex gap-12">
             {['Discord', 'X', 'Protocol'].map((item) => (
                <Link key={item} href="#" className="text-[11px] font-bold text-white/20 hover:text-white transition-all uppercase tracking-[0.4em]">{item}</Link>
             ))}
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto pt-20 mt-20 border-t border-white/[0.02] text-center">
           <span className="text-white/[0.05] text-[11px] font-bold uppercase tracking-[0.8em]">© 2026 THE ARCHITECT • SOVEREIGN ASSET ENCLAVE</span>
        </div>
      </footer>
    </div>
  );
}
