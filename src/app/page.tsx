'use client';

import { Shield, Zap, Lock, Star, CheckCircle2, ArrowRight, Menu, X, Smile, Users, Heart } from 'lucide-react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';
import LineIllustration from '@/components/LineIllustration';
import RevolvingGlobe from '@/components/RevolvingGlobe';
import SovereignHero from '@/components/SovereignHero';

// Lemonade-Standard Spring Configurations
const lemonSpring = { type: "spring", stiffness: 400, damping: 30, mass: 1 } as any;

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { ...lemonSpring }
  }
};

const staggerContainer = {
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
  const [isMounted, setIsMounted] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return <div className="min-h-screen bg-white" />;

  return (
    <div className="min-h-screen bg-white text-[#111111] overflow-x-hidden selection:bg-[#FF0066]/20">
      
      {/* Navigation: Sticky & Persistent */}
      <nav className="sticky-nav px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2 group cursor-pointer">
            <div className="w-10 h-10 bg-lemon rounded-xl flex items-center justify-center group-hover:rotate-12 transition-transform">
              <Shield className="text-white w-5 h-5" />
            </div>
            <span className="text-2xl font-black tracking-tighter">SANFA</span>
          </div>

          <div className="hidden md:flex items-center gap-10">
            {['Protection', 'How it Works', 'Reviews'].map((item) => (
              <Link key={item} href={`#${item.toLowerCase().replace(/ /g, '-')}`} className="text-sm font-bold text-gray-500 hover:text-black transition-colors">{item}</Link>
            ))}
            <Link href="/login" className="text-sm font-bold text-gray-400 hover:text-black transition-colors pl-6 border-l">Sign In</Link>
            <Link href="/login" className="btn-lemon py-3 px-10 text-sm">Check Our Prices</Link>
          </div>

          <button className="md:hidden" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
            {isMobileMenuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </nav>

      <main>
        {/* Hero Section: Centered & Massive */}
        <section className="py-24 md:py-40 px-6 container mx-auto text-center overflow-hidden">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={staggerContainer}
            className="max-w-4xl mx-auto"
          >
            <motion.h1 variants={fadeInUp} className="text-[var(--font-size-3xl)] mb-8">
              Forget everything you know about <span className="text-lemon">protection.</span>
            </motion.h1>
            
            <motion.p variants={fadeInUp} className="text-xl md:text-2xl text-gray-500 mb-12 max-w-2xl mx-auto leading-relaxed">
              Instant protection for your digital assets. No paperwork, no hassle, just pure mathematical sovereignty.
            </motion.p>

            <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-20">
              <Link href="/login" className="btn-lemon text-lg px-12 py-6 w-full sm:w-auto">
                Protect My Assets
              </Link>
              <div className="flex items-center gap-2 text-gray-400 font-bold text-sm">
                <CheckCircle2 className="w-5 h-5 text-green-500" />
                Takes 90 seconds
              </div>
            </motion.div>

            <motion.div 
              variants={fadeInUp} 
              className="relative mt-20"
              animate={{ 
                y: [0, -15, 0],
              }}
              transition={{ 
                duration: 6, 
                repeat: Infinity, 
                ease: "easeInOut" 
              }}
            >
               <SovereignHero />
            </motion.div>
          </motion.div>
        </section>

        {/* Social Proof: Trust Banner */}
        <motion.section 
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          variants={staggerContainer}
          className="py-12 border-y border-gray-50 bg-[#FAFAFA]"
        >
          <div className="container mx-auto px-6 overflow-hidden text-center">
            <motion.p variants={fadeInUp} className="text-[10px] uppercase tracking-[0.4em] font-black text-gray-400 mb-10">Trusted and featured by the industry elite</motion.p>
            <motion.div variants={fadeInUp} className="flex flex-wrap justify-center items-center gap-12 md:gap-24 opacity-30 grayscale transition-all hover:grayscale-0">
               {['Forbes', 'Wired', 'The Verge', 'TechCrunch', 'Bloomberg'].map((logo) => (
                 <span key={logo} className="text-2xl font-black tracking-tighter cursor-default">{logo}</span>
               ))}
            </motion.div>
          </div>
        </motion.section>

        {/* Features: Card Grid */}
        <section id="protection" className="py-32 px-6 container mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={staggerContainer}
          >
            <div className="text-center mb-24">
               <motion.h2 variants={fadeInUp} className="text-[var(--font-size-2xl)] mb-4">A new kind of <span className="text-lemon">safety.</span></motion.h2>
               <motion.p variants={fadeInUp} className="text-gray-500 text-lg">We built it to be so simple, you'll actually use it.</motion.p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
              {[
                { icon: Shield, title: 'Total Anonymity', desc: 'We don\'t need your name. We just need to protect your masterpiece.' },
                { icon: Zap, title: 'Instant Shield', desc: 'Response times under 100ms. Protection happens before the crawl ends.' },
                { icon: Lock, title: 'Kernel Lock', desc: 'Advanced frequency-domain interference that makes duplication impossible.' }
              ].map((feature, i) => (
                <motion.div
                  key={i}
                  variants={fadeInUp}
                  className="lemon-card p-12 flex flex-col items-center text-center"
                >
                  <div className="w-16 h-16 bg-gray-50 rounded-2xl flex items-center justify-center mb-8">
                     <feature.icon className="text-lemon w-8 h-8" />
                  </div>
                  <h3 className="text-2xl font-black mb-4">{feature.title}</h3>
                  <p className="text-gray-500 leading-relaxed">{feature.desc}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </section>

        {/* How it Works: Animation Section */}
        <section id="how-it-works" className="py-32 px-6 bg-[#FAFAFA]">
          <div className="container mx-auto grid grid-cols-1 lg:grid-cols-2 gap-24 items-center">
            <div className="order-2 lg:order-1">
               <span className="text-lemon font-black text-[10px] uppercase tracking-[0.6em] block mb-6">Built for Creators</span>
               <h2 className="text-[var(--font-size-2xl)] mb-8">Simple as <br /> <span className="text-lemon">Plug and Protect.</span></h2>
               <div className="space-y-10">
                  {[
                    { step: '01', title: 'Connect Your Enclave', desc: 'Import your assets with a single click. We handle the technical layer.' },
                    { step: '02', title: 'Deploy the Shield', desc: 'Choose your protection intensity from basic to total invisibility.' },
                    { step: '03', title: 'Rest Easy', desc: 'Monitor your asset status in real-time while we fight the scrapers.' }
                  ].map((step, i) => (
                    <div key={i} className="flex gap-8 group">
                       <span className="text-lemon/20 font-black text-6xl group-hover:text-lemon/40 transition-colors leading-none">{step.step}</span>
                       <div>
                          <h4 className="text-xl font-black mb-2">{step.title}</h4>
                          <p className="text-gray-500">{step.desc}</p>
                       </div>
                    </div>
                  ))}
               </div>
            </div>
            <div className="order-1 lg:order-2">
               <div className="bg-white p-12 rounded-[40px] shadow-2xl relative overflow-hidden group">
                  <div className="absolute top-0 right-0 p-8 flex gap-2">
                     <span className="w-2 h-2 rounded-full bg-red-400" />
                     <span className="w-2 h-2 rounded-full bg-yellow-400" />
                     <span className="w-2 h-2 rounded-full bg-green-400" />
                  </div>
                  <LineIllustration className="scale-110 group-hover:scale-125 transition-transform duration-1000" />
                  <div className="mt-8 pt-8 border-t flex items-center justify-between">
                     <div className="flex items-center gap-3">
                        <div className="w-3 h-3 bg-lemon rounded-full animate-pulse" />
                        <span className="text-[10px] font-black uppercase tracking-widest text-gray-400">Live Protection</span>
                     </div>
                     <ArrowRight className="text-lemon w-6 h-6" />
                  </div>
               </div>
            </div>
          </div>
        </section>

        {/* Reviews Section */}
        <section id="reviews" className="py-32 px-6 container mx-auto">
          <div className="text-center mb-24">
             <div className="flex justify-center gap-1 mb-8">
               {[1, 2, 3, 4, 5].map((s) => (
                 <Star key={s} className="w-8 h-8 fill-lemon text-lemon" />
               ))}
             </div>
             <h2 className="text-[var(--font-size-2xl)]">Rated 4.9/5 by over <br /> <span className="text-lemon">10,000 creators.</span></h2>
          </div>

          <div className="flex overflow-hidden group">
             <div className="flex animate-scroll group-hover:pause-animation gap-8 pr-8">
               {[
                 { user: 'CreativeDir', text: 'Literally saved our entire portfolio from scraper exhaustion.' },
                 { user: 'NexaStudio', text: 'The interface is so clean it actually feels like a relief to use.' },
                 { user: 'Solis.eth', text: 'Mathematical sovereignty is the only way forward. SANFA is it.' },
                 { user: 'HyperGlow', text: 'No paperwork. Just protection. Exactly what we needed.' },
                 { user: 'EliteAI', text: 'Irony: An AI firm using SANFA to protect their internal models. It works.' },
                 // Duplicate for infinite scroll effect
                 { user: 'CreativeDir-2', text: 'Literally saved our entire portfolio from scraper exhaustion.' },
                 { user: 'NexaStudio-2', text: 'The interface is so clean it actually feels like a relief to use.' },
                 { user: 'Solis.eth-2', text: 'Mathematical sovereignty is the only way forward. SANFA is it.' }
               ].map((review, i) => (
                 <div key={i} className="lemon-card p-10 min-w-[350px] flex flex-col justify-between">
                    <p className="text-lg font-medium mb-8">"{review.text}"</p>
                    <div className="flex items-center gap-4">
                       <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                          <Smile className="text-lemon w-6 h-6" />
                       </div>
                       <span className="font-bold text-sm">@{review.user}</span>
                    </div>
                 </div>
               ))}
             </div>
          </div>
        </section>

        {/* Global Reach: The World Shield */}
        <section className="py-32 px-6 bg-[#FFFFFF] overflow-hidden">
          <div className="container mx-auto grid grid-cols-1 lg:grid-cols-2 gap-24 items-center">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={staggerContainer}
              className="order-2 lg:order-1"
            >
               <motion.span variants={fadeInUp} className="text-lemon font-black text-[10px] uppercase tracking-[0.6em] block mb-6">Worldwide Security</motion.span>
               <motion.h2 variants={fadeInUp} className="text-[var(--font-size-2xl)] mb-8">One enclave. <br /> <span className="text-lemon">Infinite scale.</span></motion.h2>
               <motion.p variants={fadeInUp} className="text-xl text-gray-500 mb-12 max-w-lg leading-relaxed">
                  Sanfa is a decentralized global protocol. From London to Tokyo, our cryptographic shield protects millions of assets in real-time, regardless of territory.
               </motion.p>
               <motion.div variants={fadeInUp} className="flex gap-12">
                  <div>
                     <span className="text-4xl font-black block mb-2">180+</span>
                     <span className="text-gray-400 text-xs uppercase tracking-widest font-bold">Countries</span>
                  </div>
                  <div>
                     <span className="text-4xl font-black block mb-2">99.9%</span>
                     <span className="text-gray-400 text-xs uppercase tracking-widest font-bold">Uptime</span>
                  </div>
               </motion.div>
            </motion.div>
            <div className="order-1 lg:order-2 flex justify-center">
               <RevolvingGlobe />
            </div>
          </div>
        </section>

        {/* CTA Banner */}
        <section className="py-32 px-6">
           <div className="max-w-7xl mx-auto bg-lemon rounded-[60px] p-20 text-center text-white relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-full h-full bg-white/5 scale-x-0 group-hover:scale-x-100 transition-transform duration-1000 origin-left" />
              <div className="relative z-10 flex flex-col items-center">
                 <h2 className="text-[var(--font-size-2xl)] mb-8 text-white">Join the <span className="underline decoration-white/20">Sovereignty.</span></h2>
                 <p className="text-xl md:text-2xl text-white/80 mb-12 max-w-xl mx-auto">It takes only 90 seconds. Protect your masterpiece before it's too late.</p>
                 <Link href="/login" className="bg-white text-lemon px-16 py-8 rounded-full text-2xl font-black hover:scale-105 active:scale-95 transition-all shadow-2xl">
                    Get Protected Now
                 </Link>
              </div>
           </div>
        </section>
      </main>

      {/* Footer: Charcoal Minimalist */}
      <footer className="bg-[#222222] text-white py-32 px-6">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-20 mb-20">
            <div className="col-span-1 md:col-span-1">
               <div className="flex items-center gap-2 mb-8">
                  <div className="w-10 h-10 bg-lemon rounded-xl flex items-center justify-center">
                    <Shield className="text-white w-5 h-5" />
                  </div>
                  <span className="text-2xl font-black tracking-tighter">SANFA</span>
               </div>
               <p className="text-gray-400 text-sm leading-relaxed mb-10">
                  Protecting human genius in the age of neural networks. Simple. Secure. Mathematical.
               </p>
               <div className="flex gap-6">
                  <Users className="text-gray-500 hover:text-lemon cursor-pointer" />
                  <Heart className="text-gray-500 hover:text-lemon cursor-pointer" />
               </div>
            </div>
            
            {['Product', 'Company', 'Legal'].map((col) => (
              <div key={col}>
                 <h5 className="font-black uppercase text-xs tracking-widest text-gray-500 mb-8">{col}</h5>
                 <ul className="space-y-4">
                    {['About Us', 'Protection Plan', 'Manifesto', 'Privacy Policy'].map((link) => (
                      <li key={link}>
                         <Link href="#" className="text-gray-400 hover:text-white transition-colors text-sm font-bold">{link}</Link>
                      </li>
                    ))}
                 </ul>
              </div>
            ))}
          </div>
          
          <div className="pt-20 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-10">
             <span className="text-gray-600 text-[10px] font-black uppercase tracking-[0.4em]">© 2026 SANFA PROTOCOL • HUMAN S-TIER ORIGINS</span>
             <div className="flex items-center gap-1">
                <span className="text-gray-600 text-xs">Built with</span>
                <Heart className="text-lemon w-3 h-3 fill-lemon" />
                <span className="text-gray-600 text-xs">for creators.</span>
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
          animation: scroll 40s linear infinite;
        }
        .pause-animation {
          animation-play-state: paused;
        }
      `}</style>
    </div>
  );
}
