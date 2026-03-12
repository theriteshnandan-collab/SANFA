'use client';

import { Shield, Sparkles, Zap, Lock, Globe, Activity, Cpu, ArrowRight, ShieldCheck, Fingerprint, Eye, ZapOff } from 'lucide-react';
import Link from 'next/link';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';

const fadeInUp: any = {
  hidden: { opacity: 0, y: 30 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] }
  }
};

const staggerContainer: any = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2
    }
  }
};

export default function Home() {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.95]);

  return (
    <div ref={containerRef} className="min-h-screen bg-[#0a0a0b] selection:bg-gold-500/30 selection:text-white overflow-x-hidden font-sans">
      {/* Dynamic Background Elements */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] noise z-[100]" />
      <div className="fixed top-0 left-1/2 -translate-x-1/2 w-full h-full max-w-[1400px] pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-gold-500/10 blur-[120px] rounded-full animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/5 blur-[120px] rounded-full" />
      </div>

      {/* Navigation */}
      <motion.nav 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        className="px-8 py-6 flex items-center justify-between fixed top-0 w-full z-[110] glass-card border-b border-white/5 rounded-none"
      >
        <div className="flex items-center gap-4 group">
          <div className="w-10 h-10 bg-gold-500/20 rounded-xl flex items-center justify-center border border-gold-500/30 group-hover:scale-110 transition-all duration-500 shadow-xl shadow-gold-500/10 relative overflow-hidden">
            <div className="absolute inset-0 bg-gold-500/10 translate-y-full group-hover:translate-y-0 transition-transform duration-500" />
            <Shield className="text-gold-500 w-6 h-6 relative z-10" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-serif tracking-tight text-white leading-none">SANFA <span className="text-gold-500">CLOUD</span></span>
            <span className="text-[9px] uppercase tracking-[0.4em] text-white/30 font-bold mt-1">Sovereignty Protocol</span>
          </div>
        </div>
        <div className="flex items-center gap-10">
          <div className="hidden md:flex items-center gap-8">
            {['Architecture', 'Manifesto', 'Security'].map((item) => (
              <Link key={item} href={`#${item.toLowerCase()}`} className="text-[10px] font-bold text-white/30 hover:text-white transition-all uppercase tracking-[0.2em]">{item}</Link>
            ))}
          </div>
          <div className="flex items-center gap-4">
            <Link href="/login" className="text-[10px] font-bold text-white/50 hover:text-white transition-all tracking-[0.2em] uppercase hidden sm:block">Access Terminal</Link>
            <Link href="/login" className="bg-white text-black px-8 py-3 rounded-full text-xs font-black uppercase tracking-widest hover:bg-gold-500 transition-all shadow-2xl hover:shadow-gold-500/40">Engage</Link>
          </div>
        </div>
      </motion.nav>

      <main className="relative z-10">
        {/* HERO SECTION: The Sovereignty Entry */}
        <section className="relative min-h-screen flex flex-col items-center justify-center pt-32 pb-20 px-8">
          <motion.div 
            style={{ opacity, scale }}
            variants={staggerContainer as any}
            initial="hidden"
            animate="visible"
            className="max-w-7xl mx-auto text-center"
          >
            <motion.div variants={fadeInUp as any} className="inline-flex items-center gap-3 px-6 py-2 rounded-full bg-gold-500/10 border border-gold-500/20 mb-10 shadow-lg shadow-gold-500/5">
              <Sparkles className="w-4 h-4 text-gold-500 animate-pulse" />
              <span className="text-[10px] font-black text-gold-400 uppercase tracking-[0.4em]">Engine V5.2 Fully Operational</span>
            </motion.div>

            <motion.h1 
              variants={fadeInUp as any}
              className="text-[var(--font-size-4xl)] font-serif mb-10 tracking-tighter leading-[0.85] text-gradient py-2"
            >
              Defend the <br /> <span className="text-gold-500 italic">Human Genius.</span>
            </motion.h1>

            <motion.p 
              variants={fadeInUp as any}
              className="text-white/40 max-w-3xl mx-auto text-lg md:text-2xl font-sans leading-relaxed mb-16 tracking-tight"
            >
              SANFA is the world's most advanced deterrent against AI scraping. We mathematically poison your artwork to render it <span className="text-white font-bold italic">invisible to neural extractors</span> while preserving every pixel of human intent.
            </motion.p>

            <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row items-center justify-center gap-8">
              <Link href="/login" className="group bg-white text-black px-14 py-6 rounded-full text-lg font-black uppercase tracking-widest hover:bg-gold-500 transition-all flex items-center gap-4 shadow-[0_0_50px_rgba(255,255,255,0.1)] hover:shadow-gold-500/50">
                Engage Protection
                <ArrowRight className="w-5 h-5 group-hover:translate-x-2 transition-transform" />
              </Link>
              <Link href="#manifesto" className="text-[11px] font-black uppercase tracking-[0.4em] text-white/30 hover:text-white transition-all flex items-center gap-3 border-b border-white/10 pb-1 hover:border-gold-500">
                The Manifesto
              </Link>
            </motion.div>
          </motion.div>

          {/* Abstract Engine Visual */}
          <motion.div 
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 1.5, ease: [0.22, 1, 0.36, 1] }}
            className="w-full max-w-5xl mt-32 aspect-video glass-card rounded-[48px] overflow-hidden relative group"
          >
            <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a0b] via-transparent to-transparent z-10" />
            <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&q=80&w=2000')] bg-cover bg-center grayscale opacity-20 group-hover:scale-110 transition-transform duration-[10s]" />
            <div className="absolute inset-0 flex items-center justify-center z-20">
              <div className="flex flex-col items-center gap-6">
                <div className="w-24 h-24 bg-gold-500/10 rounded-full flex items-center justify-center border border-gold-500/30 scale-150 animate-pulse">
                  <ShieldCheck className="text-gold-500 w-12 h-12" />
                </div>
                <span className="text-[10px] font-black uppercase tracking-[0.5em] text-gold-500">Neural Tunnel Enabled</span>
              </div>
            </div>
            {/* Grid Overlay */}
            <div className="absolute inset-0 z-10 opacity-20" style={{ backgroundImage: 'radial-gradient(rgba(201,168,76,0.2) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
          </motion.div>
        </section>

        {/* ARCHITECTURE: The Decoupled Cloud Pipeline */}
        <section id="architecture" className="py-40 relative border-y border-white/5 bg-white/[0.01]">
          <div className="max-w-7xl mx-auto px-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-32 items-center">
              <motion.div 
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-100px" }}
                variants={staggerContainer as any}
              >
                <motion.span variants={fadeInUp as any} className="text-gold-500 font-black text-[10px] uppercase tracking-[0.5em] block mb-6">Scale & Sovereignty</motion.span>
                <motion.h2 variants={fadeInUp as any} className="text-[var(--font-size-3xl)] font-serif mb-10 leading-[0.9] tracking-tighter">Decoupled <br /> <span className="text-white/30">GPU Core.</span></motion.h2>
                <motion.p variants={fadeInUp as any} className="text-white/40 text-xl mb-14 font-sans leading-relaxed tracking-tight">
                  Our infrastructure is designed for high-frequency protection. By decoupling data ingestion from the adversarial cluster, we achieve 100% uptime and sub-second job handoffs. When you upload, our serverless enclaves immediately wage war on feature extraction.
                </motion.p>
                
                <div className="space-y-10">
                  {[
                    { icon: Globe, title: 'Global Edge Ingestion', desc: 'Direct-to-bucket pre-signed uploads bypass every server bottleneck.' },
                    { icon: Cpu, title: 'Serverless H100 Cluster', desc: 'On-demand GPU workers that spin up in milliseconds for infinite scale.' },
                    { icon: Activity, title: 'Live Neural Telemetry', desc: 'Watch the adversarial distance grow in real-time as the poison takes root.' }
                  ].map((item, i) => (
                    <motion.div 
                      key={i}
                      variants={fadeInUp as any}
                      whileHover={{ x: 15 }}
                      className="flex items-start gap-8 group"
                    >
                      <div className="w-14 h-14 rounded-2xl bg-gold-500/10 flex items-center justify-center border border-gold-500/20 group-hover:border-gold-500/50 transition-all shadow-xl shadow-gold-500/5">
                        <item.icon className="text-gold-500 w-6 h-6" />
                      </div>
                      <div>
                        <h4 className="font-serif text-2xl mb-2 text-white/90 group-hover:text-gold-500 transition-colors">{item.title}</h4>
                        <p className="text-white/30 text-base font-sans leading-relaxed">{item.desc}</p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>

              {/* BENTO GRID VISUALIZATION */}
              <div className="grid grid-cols-2 gap-6 relative">
                {/* Glow behind grid */}
                <div className="absolute inset-0 bg-gold-500/5 blur-[100px] rounded-full" />
                
                <div className="space-y-6">
                  <motion.div 
                    whileHover={{ y: -10 }}
                    className="h-56 glass-card rounded-[40px] p-10 flex flex-col justify-end border-gold-500/10"
                  >
                    <span className="text-gold-500 font-serif text-5xl mb-3">99.9<span className="text-2xl text-gold-500/40">%</span></span>
                    <span className="text-white/30 text-[10px] font-black uppercase tracking-[0.3em]">Processing Reliability</span>
                  </motion.div>
                  <motion.div 
                    whileHover={{ y: -10 }}
                    className="h-80 glass-card rounded-[40px] p-10 bg-gold-500/5 border-gold-500/30 flex flex-col items-center justify-center text-center"
                  >
                    <Zap className="text-gold-500 w-16 h-16 mb-8 animate-bounce" />
                    <span className="text-white font-serif text-3xl mb-3">200K+</span>
                    <span className="text-white/30 text-[10px] font-black uppercase tracking-[0.3em]">Monthly Scrambles</span>
                  </motion.div>
                </div>
                <div className="space-y-6 pt-12">
                  <motion.div 
                    whileHover={{ y: -10 }}
                    className="h-80 glass-card rounded-[40px] p-10 overflow-hidden relative group border-white/5"
                  >
                    <div className="absolute inset-0 bg-gold-500/10 translate-y-full group-hover:translate-y-0 transition-transform duration-700" />
                    <div className="relative z-20 h-full flex flex-col">
                      <Fingerprint className="text-white w-12 h-12 mb-8" />
                      <span className="text-white font-serif text-3xl mb-3">Immutable Hash</span>
                      <span className="text-white/30 text-[10px] font-black uppercase tracking-[0.3em]">C2PA Authenticated</span>
                    </div>
                  </motion.div>
                  <motion.div 
                    whileHover={{ y: -10 }}
                    className="h-56 glass-card rounded-[40px] p-10 flex flex-col justify-center border-gold-500/10"
                  >
                    <div className="flex items-center gap-3 mb-4">
                       <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping" />
                       <span className="text-emerald-500 text-[10px] font-black uppercase tracking-widest">Active nodes</span>
                    </div>
                    <span className="text-white font-serif text-2xl">6.4ms Latency</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* SECURITY: The Shield Logic */}
        <section id="security" className="py-40 bg-[#070708]">
          <div className="max-w-7xl mx-auto px-8 text-center mb-24">
            <span className="text-gold-500 font-black text-[10px] uppercase tracking-[0.5em] block mb-6">The Protocol</span>
            <h2 className="text-[var(--font-size-3xl)] font-serif text-gradient leading-none">Silent Warfare.</h2>
          </div>

          <div className="max-w-7xl mx-auto px-8 grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { icon: Eye, title: 'Neural Collapse', desc: 'We identify the high-probability feature kernels AI models use for recognition and systematically invert their signals.' },
              { icon: Lock, title: 'Frequency Scramble', desc: 'Our Engine V5 adds invisible mathematical noise to the spectral domain, creating a "black hole" in the model\'s memory.' },
              { icon: ZapOff, title: 'Zero Training Decay', desc: 'Any model trained on protected assets suffers from catastrophic forgetting, protecting not just one image, but your entire style.' }
            ].map((item, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2 }}
                variants={fadeInUp as any}
                className="glass-card rounded-[40px] p-12 border-white/5 hover:border-gold-500/30 transition-all flex flex-col items-center text-center group"
              >
                <div className="w-20 h-20 bg-white/5 rounded-[32px] flex items-center justify-center mb-10 border border-white/10 group-hover:bg-gold-500/10 group-hover:border-gold-500/30 transition-all">
                  <item.icon className="text-white/40 w-10 h-10 group-hover:text-gold-500 transition-colors" />
                </div>
                <h4 className="text-3xl font-serif mb-6 text-white leading-tight">{item.title}</h4>
                <p className="text-white/30 text-lg leading-relaxed font-sans">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* MANIFESTO: Digital Sovereignty */}
        <section id="manifesto" className="py-40">
          <div className="max-w-5xl mx-auto px-8">
            <div className="glass-card rounded-[64px] p-20 border-gold-500/10 relative overflow-hidden">
               {/* Background Logo */}
               <Shield className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-gold-500/5 w-[600px] h-[600px] rotate-12 pointer-events-none" />
               
               <div className="relative z-10">
                 <h2 className="text-[var(--font-size-2xl)] font-serif mb-20 text-center tracking-tighter">The Digital Sovereignty <br /> <span className="text-gold-500 uppercase font-sans font-black text-sm tracking-[0.5em] block mt-4">Manifesto</span></h2>
                 
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-20">
                    <div className="space-y-12">
                       <div className="space-y-4">
                          <span className="text-gold-500 font-serif text-4xl italic">01.</span>
                          <h3 className="text-white text-2xl font-serif">Consent is Non-Negotiable.</h3>
                          <p className="text-white/40 leading-relaxed text-lg font-sans">AI treats the web as a buffet. We believe every pixel is a property. Learning should require permission, or at the very least, a decisive technical cost.</p>
                       </div>
                       <div className="space-y-4">
                          <span className="text-gold-500 font-serif text-4xl italic">02.</span>
                          <h3 className="text-white text-2xl font-serif">The Right to Confusion.</h3>
                          <p className="text-white/40 leading-relaxed text-lg font-sans">Confusion is the ultimate privacy. By injecting "Gradient Poison" into image artifacts, SANFA creators strike back at mass-scraping robots.</p>
                       </div>
                    </div>
                    <div className="flex flex-col justify-center items-center p-12 bg-gold-500/5 rounded-[48px] border border-gold-500/10">
                       <p className="text-gold-500 font-serif text-3xl italic text-center leading-tight mb-10">
                          "Your genius is not training data. <br /> It is your heritage."
                       </p>
                       <Link href="/login" className="bg-white text-black px-12 py-5 rounded-full text-sm font-black uppercase tracking-[0.3em] hover:bg-gold-500 transition-all">Engage the Shield</Link>
                    </div>
                 </div>
               </div>
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="py-24 border-t border-white/5 relative bg-[#070708]">
        <div className="max-w-7xl mx-auto px-8 flex flex-col md:flex-row items-center justify-between gap-12">
          <div className="flex flex-col items-center md:items-start gap-4">
            <div className="flex items-center gap-4">
              <Shield className="text-gold-500/50 w-6 h-6" />
              <span className="text-white font-serif text-xl tracking-tight">SANFA PROTOCOL</span>
            </div>
            <p className="text-white/20 text-xs font-sans max-w-xs text-center md:text-left leading-relaxed">
              Establishing a decentralized defense for the future of human creativity. 
              Built for the sovereign artist.
            </p>
          </div>
          
          <div className="flex flex-col items-center gap-6">
            <span className="text-gold-500 font-black text-[10px] uppercase tracking-[0.5em]">Global Statistics</span>
            <div className="flex items-center gap-12">
               <div className="text-center">
                  <span className="text-white font-serif text-2xl block">14.5M</span>
                  <span className="text-white/20 text-[9px] font-black uppercase tracking-widest">Assets Shielded</span>
               </div>
               <div className="text-center">
                  <span className="text-white font-serif text-2xl block">182K</span>
                  <span className="text-white/20 text-[9px] font-black uppercase tracking-widest">Artists Active</span>
               </div>
            </div>
          </div>

          <div className="flex items-center gap-10">
            {['Twitter', 'Discord', 'Docs'].map((item) => (
              <Link key={item} href="#" className="text-white/30 hover:text-white transition-colors text-[10px] font-bold uppercase tracking-[0.3em]">{item}</Link>
            ))}
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-8 pt-12 mt-12 border-t border-white/[0.03] text-center">
           <span className="text-white/[0.05] text-[10px] font-black uppercase tracking-[0.5em]">© 2026 SANFA CLOUD • BY THE ARCHITECT</span>
        </div>
      </footer>
    </div>
  );
}
