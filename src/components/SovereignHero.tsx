'use client';

import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect } from 'react';

export default function SovereignHero() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Elite-fidelity spring configurations
  const springX = useSpring(mouseX, { stiffness: 50, damping: 20 });
  const springY = useSpring(mouseY, { stiffness: 50, damping: 20 });

  // Triple-layer parallax transformations
  const heroX = useTransform(springX, [-500, 500], [-30, 30]);
  const heroY = useTransform(springY, [-500, 500], [-30, 30]);
  const nodeX = useTransform(springX, [-500, 500], [-50, 50]);
  const nodeY = useTransform(springY, [-500, 500], [-50, 50]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      mouseX.set(clientX - innerWidth / 2);
      mouseY.set(clientY - innerHeight / 2);
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="relative w-full max-w-4xl mx-auto aspect-[4/3] flex items-center justify-center overflow-visible cursor-default group">
      
      {/* Background Glow Bloom (Managed Properly) */}
      <div className="absolute w-[100%] h-[100%] bg-gradient-to-br from-lemon/5 via-white to-transparent rounded-full filter blur-[150px] pointer-events-none animate-pulse" />

      {/* Layer 1: The High-Fidelity Masterpiece (Restored Girl) */}
      <motion.div
        className="relative z-10 w-full h-full flex items-center justify-center"
        style={{ x: heroX, y: heroY }}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        <motion.img
          src="/assets/god-hero.png"
          alt="Sovereign Protection"
          className="w-full h-full object-contain filter drop-shadow-[0_60px_150px_rgba(255,0,102,0.1)]"
          animate={{ 
            y: [0, -15, 0],
          }}
          transition={{ 
            duration: 8, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Security Shield (Geometric & Clean - No hair-like curves) */}
        <motion.div 
          className="absolute inset-x-[-5%] inset-y-[-5%] border-[0.8px] border-lemon/20 rounded-[80px] pointer-events-none z-0"
          animate={{ 
            scale: [1, 1.02, 1],
            opacity: [0.3, 0.6, 0.3]
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>

      {/* Layer 2: Dynamic Security Nodes (Premium Parallax) */}
      <motion.div style={{ x: nodeX, y: nodeY }} className="absolute inset-0 pointer-events-none z-20">
        {[
          { top: '20%', left: '15%', delay: 0 },
          { top: '80%', left: '85%', delay: 2 },
          { top: '10%', left: '75%', delay: 4 },
          { top: '70%', left: '10%', delay: 6 },
        ].map((node, i) => (
          <motion.div
            key={i}
            className="absolute w-4 h-4 rounded-full"
            style={{ 
              top: node.top, 
              left: node.left,
              backgroundColor: '#FF0066',
              boxShadow: '0 0 30px rgba(255, 0, 102, 0.7)'
            }}
            animate={{ 
              scale: [1, 1.5, 1],
              opacity: [0.4, 0.9, 0.4],
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity, 
              delay: node.delay,
            }}
          />
        ))}
      </motion.div>
    </div>
  );
}
