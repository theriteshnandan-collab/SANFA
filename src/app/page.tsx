'use client';

import { Shield, Zap, Lock, Star, CheckCircle2, ArrowRight, Menu, X, Smile, Users, Heart, Cpu, Globe, Activity } from 'lucide-react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import LineIllustration from '@/components/LineIllustration';
import RevolvingGlobe from '@/components/RevolvingGlobe';
import SovereignHero from '@/components/SovereignHero';

// Master-Craft Spring Configurations
const masterSpring = { type: "spring", stiffness: 350, damping: 35, mass: 1 } as any;

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { ...masterSpring }
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
      delayChildren: 0.1
    }
  }
};

export default function Home() {
  const [isMounted, setIsMounted] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return <div className="min-h-screen bg-white" />;

  return (
    <div className="min-h-screen bg-white text-[#111111] overflow-x-hidden selection:bg-[#FF0066]/20 font-sans">
      
      {/* Navigation: Sticky & Elite */}
      <nav className="sticky-nav px-8 py-6">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3 group cursor-pointer">
            <div className="w-12 h-12 bg-lemon rounded-2xl flex items-center justify-center group-hover:rotate-12 transition-transform shadow-lg shadow-lemon/10">
              <Shield className="text-white w-6 h-6" />
            </div>
            <span className="text-3xl font-black tracking-tighter">SANFA</span>
          </div>

          <div className="hidden lg:flex items-center gap-12">
            {['Protection', 'Protocol', 'Reviews'].map((item) => (
              <Link key={item} href={`#${item.toLowerCase().replace(/ /g, '-')}`} className="text-base font-black text-gray-400 hover:text-black transition-colors tracking-tight">{item}</Link>
            ))}
            <Link href="/login" className="text-base font-black text-gray-300 hover:text-black transition-colors pl-8 border-l border-gray-100 uppercase tracking-widest text-[10px]">Sign In</Link>
            <Link href="/login" className="btn-lemon py-4 px-12 text-sm">Check Individual Pricing</Link>
          </div>

          <button className="lg:hidden" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
            {isMobileMenuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </nav>

      <main>
        {/* Hero Section: The Grand Scale */}
        <section className="pt-32 pb-48 px-8 container mx-auto text-center relative">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
            className="max-w-6xl mx-auto"
          >
            <motion.div variants={fadeInUp} className="mb-6 inline-flex items-center gap-2 bg-gray-50 border border-gray-100 rounded-full px-4 py-2">
                 <div className="w-2 h-2 bg-green-500 rounded-full animate-ping" />
                 <span className="text-[10px] font-black uppercase tracking-[0.3em] text-gray-400">System Status: Sovereign & Secure</span>
            </motion.div>

            <motion.h1 variants={fadeInUp} className="text-[var(--font-size-3xl)] mb-10 tracking-[-0.05em]">
              The end of <span className="text-lemon">digital vulnerability.</span>
            </motion.h1>
            
            <motion.p variants={fadeInUp} className="text-2xl md:text-3xl text-gray-400 mb-16 max-w-4xl mx-auto leading-tight font-medium">
              We engineered the world's first mathematical enclave. No passwords, no centralized risks—just pure, unbreakable sovereignty over your masterpiece.
            </motion.p>

            <motion.div variants={fadeInUp} className="flex flex-col md:flex-row items-center justify-center gap-8 mb-32">
              <Link href="/login" className="btn-lemon text-xl px-16 py-8 w-full md:w-auto shadow-2xl shadow-lemon/30">
                Protect My Enclave
              </Link>
              <div className="flex items-center gap-3 text-gray-400 font-black text-xs uppercase tracking-widest">
                <CheckCircle2 className="w-6 h-6 text-green-500" />
                Activation: 90 Seconds
              </div>
            </motion.div>

            <motion.div 
              variants={fadeInUp} 
              className="relative w-full max-w-5xl mx-auto scale-[1.1] md:scale-[1.2]"
            >
               <SovereignHero />
            </motion.div>
          </motion.div>
        </section>

        {/* Global Protection Section (Shifted & Bold) */}
        <section id="protection" className="py-48 px-8 bg-[#FAFAFA] overflow-hidden">
          <div className="container mx-auto grid grid-cols-1 lg:grid-cols-2 gap-32 items-center">
             
             {/* Left: Interactive Globe Shield */}
             <div className="relative flex justify-center group order-2 lg:order-1">
                <div className="absolute -top-12 left-0 z-30">
                   <motion.div 
                     animate={{ y: [0, -10, 0] }}
                     transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                     className="bg-white/80 backdrop-blur-xl border border-gray-100 p-6 rounded-3xl shadow-xl flex items-center gap-4"
                   >
                       <div className="bg-lemon/10 p-3 rounded-xl"><Lock className="text-lemon w-6 h-6" /></div>
                       <div>
                          <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-gray-400">Node Status</span>
                          <span className="font-black text-xl">ENCRYPTED</span>
                       </div>
                   </motion.div>
                </div>

                <div className="absolute -bottom-12 right-0 z-30">
                   <motion.div 
                     animate={{ y: [0, 10, 0] }}
                     transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                     className="bg-white/80 backdrop-blur-xl border border-gray-100 p-6 rounded-3xl shadow-xl flex items-center gap-4"
                   >
                       <div className="bg-lemon/10 p-3 rounded-xl"><Globe className="text-lemon w-6 h-6" /></div>
                       <div>
                          <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-gray-400">Coverage</span>
                          <span className="font-black text-xl">SECURE</span>
                       </div>
                   </motion.div>
                </div>

                <RevolvingGlobe />
             </div>

             {/* Right: The Sanfa Narrative */}
             <div className="order-1 lg:order-2">
                <span className="text-lemon font-black text-xs uppercase tracking-[0.6em] block mb-8">Protocol Engineering</span>
                <h2 className="text-[var(--font-size-2xl)] mb-10">One enclave. <br /> <span className="text-lemon">Global Protection.</span></h2>
                <p className="text-2xl text-gray-400 mb-16 leading-relaxed font-medium">
                   We don't just host; we crystallize. Sanfa transforms your digital presence into a cryptographic fortress that exists everywhere and nowhere at once.
                </p>
                
                <div className="grid grid-cols-2 gap-12">
                   <div className="group">
                      <span className="text-6xl font-black block mb-4 group-hover:text-lemon transition-colors">180+</span>
                      <span className="text-gray-400 text-[10px] uppercase tracking-[0.4em] font-black">Secure Territories</span>
                   </div>
                   <div className="group">
                      <span className="text-6xl font-black block mb-4 group-hover:text-lemon transition-colors">99.9%</span>
                      <span className="text-gray-400 text-[10px] uppercase tracking-[0.4em] font-black">Uptime Integrity</span>
                   </div>
                </div>
             </div>
          </div>
        </section>

        {/* Technical Protocol Section (Fills 'Empty' space) */}
        <section id="protocol" className="py-48 px-8 container mx-auto">
           <div className="text-center mb-32">
              <span className="text-lemon font-black text-xs uppercase tracking-[0.6em] block mb-8">The Stack</span>
              <h2 className="text-[var(--font-size-2xl)]">Programmed for <br /> <span className="text-lemon">Absolute Certainty.</span></h2>
           </div>

           <div className="grid grid-cols-1 md:grid-cols-3 gap-16">
              {[
                { 
                  icon: Cpu, 
                  title: 'Kernel Sovereignty', 
                  desc: 'Individual hardware-level isolation for every user enclave. Physical separation in a virtual world.',
                  tags: ['L1 Protection', 'OLED Ready']
                },
                { 
                  icon: Activity, 
                  title: 'Kinetic Defense', 
                  desc: 'Real-time frequency modulation to disrupt automated scrapers and malicious crawlers instantly.',
                  tags: ['Live Engine', '60 FPS']
                },
                { 
                  icon: Shield, 
                  title: 'Mathematical Proof', 
                  desc: 'Every layer of protection is verified by cryptographic proofs that cannot be argued or bypassed.',
                  tags: ['Immutable', 'S-Tier']
                }
              ].map((spec, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="bg-white border border-gray-50 p-16 rounded-[40px] hover:shadow-2xl hover:border-lemon/10 transition-all duration-500 group"
                >
                  <div className="w-20 h-20 bg-gray-50 rounded-3xl flex items-center justify-center mb-12 group-hover:scale-110 transition-transform duration-500">
                     <spec.icon className="text-lemon w-10 h-10" />
                  </div>
                  <h3 className="text-3xl font-black mb-6">{spec.title}</h3>
                  <p className="text-lg text-gray-400 mb-10 leading-relaxed">{spec.desc}</p>
                  <div className="flex gap-3">
                     {spec.tags.map(tag => (
                       <span key={tag} className="text-[10px] font-black uppercase tracking-widest text-gray-300 border border-gray-100 px-3 py-1 rounded-full">{tag}</span>
                     ))}
                  </div>
                </motion.div>
              ))}
           </div>
        </section>

        {/* Reviews: The Creator Network */}
        <section id="reviews" className="py-48 px-8 bg-[#FAFAFA]">
          <div className="text-center mb-32">
             <div className="flex justify-center gap-2 mb-10">
               {[1, 2, 3, 4, 5].map((s) => (
                 <Star key={s} className="w-10 h-10 fill-lemon text-lemon" />
               ))}
             </div>
             <h2 className="text-[var(--font-size-2xl)] tracking-tighter">Certified S-Tier Excellence.</h2>
          </div>

          <div className="flex overflow-hidden group">
             <div className="flex animate-scroll group-hover:pause-animation gap-12 pr-12">
               {[
                 { user: 'CreativeDir', text: 'Literally saved our entire portfolio from scraper exhaustion. The best.' },
                 { user: 'NexaStudio', text: 'The interface is so clean it actually feels like a relief to use every day.' },
                 { user: 'Solis.eth', text: 'Mathematical sovereignty is the only way forward. SANFA is the oracle.' },
                 { user: 'HyperGlow', text: 'No paperwork. Just protection. Exactly what the decentralized web needed.' },
                 { user: 'EliteAI', text: 'Irony: An AI firm using SANFA to protect their internal models. It just works.' },
                 { user: 'Aura.Art', text: 'The visual language says everything. Premium protection for premium assets.' }
               ].map((review, i) => (
                 <div key={i} className="bg-white p-16 rounded-[50px] min-w-[450px] shadow-sm flex flex-col justify-between border border-gray-50">
                    <p className="text-2xl font-black mb-12 leading-snug">"{review.text}"</p>
                    <div className="flex items-center gap-6">
                       <div className="w-14 h-14 bg-gray-50 rounded-full flex items-center justify-center">
                          <Smile className="text-lemon w-8 h-8" />
                       </div>
                       <div className="flex flex-col">
                          <span className="font-black text-lg tracking-tight">@{review.user}</span>
                          <span className="text-[10px] uppercase font-black tracking-widest text-gray-300">Verified Client</span>
                       </div>
                    </div>
                 </div>
               ))}
               {/* Duplication for marquee */}
               {[1,2,3].map(n => (
                  <div key={n} className="bg-white p-16 rounded-[50px] min-w-[450px] shadow-sm flex flex-col justify-between border border-gray-50 opacity-50">
                    <p className="text-2xl font-black mb-12 leading-snug">"Mathematical sovereignty is the only way forward. SANFA is the oracle."</p>
                    <div className="flex items-center gap-6">
                       <div className="w-14 h-14 bg-gray-50 rounded-full flex items-center justify-center">
                          <Smile className="text-lemon w-8 h-8" />
                       </div>
                       <span className="font-black text-lg tracking-tight">@Solis.eth</span>
                    </div>
                 </div>
               ))}
             </div>
          </div>
        </section>

        {/* CTA: Final Conquest */}
        <section className="py-48 px-8">
           <div className="max-w-[1400px] mx-auto bg-lemon rounded-[80px] p-32 text-center text-white relative overflow-hidden group shadow-2xl shadow-lemon/30">
              <div className="absolute inset-0 bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
              <div className="relative z-10 flex flex-col items-center">
                 <h2 className="text-[var(--font-size-3xl)] mb-12 text-white leading-[0.9]">Secure the <br /> <span className="underline decoration-white/30">Future.</span></h2>
                 <p className="text-2xl md:text-3xl text-white/90 mb-20 max-w-3xl mx-auto font-medium">It takes only 90 seconds to deploy the enclave. Protect your masterpiece before the crawlers find it.</p>
                 <Link href="/login" className="bg-white text-lemon px-20 py-10 rounded-full text-3xl font-black hover:scale-110 active:scale-95 transition-all shadow-white/20 shadow-2xl">
                    Get Protected - Free
                 </Link>
              </div>
           </div>
        </section>
      </main>

      {/* Footer: Charcoal Elite */}
      <footer className="bg-[#111111] text-white py-48 px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-24 mb-32">
            <div className="col-span-1 md:col-span-1">
               <div className="flex items-center gap-3 mb-10">
                  <div className="w-12 h-12 bg-lemon rounded-2xl flex items-center justify-center shadow-lg shadow-lemon/20">
                    <Shield className="text-white w-6 h-6" />
                  </div>
                  <span className="text-3xl font-black tracking-tighter">SANFA</span>
               </div>
               <p className="text-gray-400 text-lg leading-relaxed mb-12 font-medium">
                  Protecting human genius in the age of neural networks. Pure. Mathematical. Sovereign.
               </p>
               <div className="flex gap-8">
                  <Users className="text-gray-600 hover:text-lemon cursor-pointer transition-colors w-6 h-6" />
                  <Heart className="text-gray-600 hover:text-lemon cursor-pointer transition-colors w-6 h-6" />
               </div>
            </div>
            
            {['Product', 'Protocol', 'Legal'].map((col) => (
              <div key={col}>
                 <h5 className="font-black uppercase text-[10px] tracking-[0.6em] text-gray-600 mb-10">{col}</h5>
                 <ul className="space-y-6">
                    {['Protection Plan', 'Enclave Access', 'Technical Specs', 'Manifesto', 'Privacy Protocol'].map((link) => (
                      <li key={link}>
                         <Link href="#" className="text-gray-400 hover:text-white transition-colors text-base font-black tracking-tight">{link}</Link>
                      </li>
                    ))}
                 </ul>
              </div>
            ))}
          </div>
          
          <div className="pt-24 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-12">
             <span className="text-gray-600 text-[10px] font-black uppercase tracking-[0.5em]">© 2026 SANFA PROTOCOL • HUMAN S-TIER ORIGINS FOUNDATION</span>
             <div className="flex items-center gap-4">
                <span className="text-gray-600 text-xs font-black uppercase tracking-widest">Built with</span>
                <Heart className="text-lemon w-4 h-4 fill-lemon" />
                <span className="text-gray-600 text-xs font-black uppercase tracking-widest">for the People.</span>
             </div>
          </div>
        </div>
      </footer>

      {/* Animation Styles for Marquee */}
      <style jsx global>{`
        @keyframes scroll {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-scroll {
          display: flex;
          width: fit-content;
          animation: scroll 60s linear infinite;
        }
        .pause-animation {
          animation-play-state: paused;
        }
      `}</style>
    </div>
  );
}
