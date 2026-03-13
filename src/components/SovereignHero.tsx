'use client';

import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect } from 'react';

export default function SovereignHero() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // High-fidelity spring configurations
  const springX = useSpring(mouseX, { stiffness: 60, damping: 25 });
  const springY = useSpring(mouseY, { stiffness: 60, damping: 25 });

  // Triple-layer parallax transformations for 'Cooler' depth
  const heroX = useTransform(springX, [-500, 500], [-25, 25]);
  const heroY = useTransform(springY, [-500, 500], [-25, 25]);
  const bgX = useTransform(springX, [-500, 500], [20, -20]);
  const bgY = useTransform(springY, [-500, 500], [20, -20]);
  const nodeX = useTransform(springX, [-500, 500], [-40, 40]);
  const nodeY = useTransform(springY, [-500, 500], [-40, 40]);

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
    <div className="relative w-full max-w-3xl mx-auto aspect-[4/3] flex items-center justify-center overflow-visible cursor-default group">
      {/* Layer 0: Background Zen Pattern (Deep Parallax) */}
      <motion.div 
        className="absolute inset-x-[-15%] inset-y-[-15%] opacity-[0.04] pointer-events-none" 
        style={{ 
          backgroundImage: 'radial-gradient(circle, #000 1.2px, transparent 1.2px)', 
          backgroundSize: '45px 45px',
          x: bgX,
          y: bgY
        }} 
      />

      {/* Layer 1: Ambient Protective Bloom */}
      <div className="absolute w-[90%] h-[90%] bg-gradient-to-br from-[#FF0066]/5 via-white to-transparent rounded-full filter blur-[120px] pointer-events-none animate-pulse" />

      {/* Layer 2: The High-Fidelity Masterpiece (Restored Girl) */}
      <motion.div
        className="relative z-10 w-full h-full flex items-center justify-center"
        style={{ x: heroX, y: heroY }}
        initial={{ opacity: 0, scale: 0.95, y: 30 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 1.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.img
          src="/assets/god-hero.png"
          alt="Sovereign Protection - Relaxed User"
          className="w-full h-full object-contain filter drop-shadow-[0_45px_120px_rgba(255,0,102,0.12)]"
          animate={{ 
            y: [0, -25, 0],
          }}
          transition={{ 
            duration: 12, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Layer 3: Dynamic Security Nodes (Parallax Bloom) */}
        <motion.div style={{ x: nodeX, y: nodeY }} className="absolute inset-0 pointer-events-none">
          {[
            { top: '25%', left: '20%', delay: 0 },
            { top: '75%', left: '80%', delay: 2 },
            { top: '15%', left: '70%', delay: 4 },
            { top: '60%', left: '15%', delay: 6 },
          ].map((node, i) => (
            <motion.div
              key={i}
              className="absolute w-3 h-3 rounded-full"
              style={{ 
                top: node.top, 
                left: node.left,
                backgroundColor: '#FF0066',
                boxShadow: '0 0 25px rgba(255, 0, 102, 0.6)'
              }}
              animate={{ 
                scale: [1, 1.6, 1],
                opacity: [0.3, 1, 0.3],
                filter: ['blur(1px)', 'blur(3px)', 'blur(1px)']
              }}
              transition={{ 
                duration: 5, 
                repeat: Infinity, 
                delay: node.delay,
                ease: "easeInOut" 
              }}
            />
          ))}
        </motion.div>

        {/* Ambient Orbiting Ring */}
        <motion.div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[95%] h-[95%] border-[0.6px] border-[#FF0066]/15 rounded-full"
          animate={{ 
            rotateZ: 360,
            scale: [1, 1.02, 1],
          }}
          transition={{ 
            rotateZ: { duration: 60, repeat: Infinity, ease: "linear" },
            scale: { duration: 10, repeat: Infinity, ease: "easeInOut" }
          }}
        />
      </motion.div>
      
      {/* Premium Silk Overlay */}
      <div className="absolute inset-0 bg-white/[0.01] backdrop-blur-[1px] pointer-events-none rounded-[100px]" />
    </div>
  );
}
