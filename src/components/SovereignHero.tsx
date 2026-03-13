'use client';

import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect } from 'react';

export default function SovereignHero() {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Smooth springs for cursor parallax
  const springX = useSpring(mouseX, { stiffness: 100, damping: 30 });
  const springY = useSpring(mouseY, { stiffness: 100, damping: 30 });

  // Parallax transformations
  const heroX = useTransform(springX, [-500, 500], [-15, 15]);
  const heroY = useTransform(springY, [-500, 500], [-15, 15]);
  const bgX = useTransform(springX, [-500, 500], [10, -10]);
  const bgY = useTransform(springY, [-500, 500], [10, -10]);

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
    <div className="relative w-full max-w-2xl mx-auto aspect-[4/3] flex items-center justify-center overflow-hidden cursor-default group">
      {/* Background Zen Pattern (Parallax) */}
      <motion.div 
        className="absolute inset-x-[-10%] inset-y-[-10%] opacity-[0.05] pointer-events-none" 
        style={{ 
          backgroundImage: 'radial-gradient(circle, #000 1px, transparent 1px)', 
          backgroundSize: '40px 40px',
          x: bgX,
          y: bgY
        }} 
      />

      <motion.div
        className="relative z-10 w-full h-full flex items-center justify-center pointer-events-none"
        style={{ x: heroX, y: heroY }}
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {/* The God-Tier Illustration */}
        <motion.img
          src="/assets/god-hero.png"
          alt="Secure and Relaxed with Sanfa"
          className="w-full h-full object-contain drop-shadow-[0_40px_100px_rgba(255,0,102,0.08)]"
          animate={{ 
            y: [0, -20, 0],
          }}
          transition={{ 
            duration: 10, 
            repeat: Infinity, 
            ease: "easeInOut" 
          }}
        />

        {/* Dynamic Glimmer Nodes (Electronic Pulse) */}
        {[
          { top: '22%', left: '18%', delay: 0 },
          { top: '78%', left: '82%', delay: 1.5 },
          { top: '15%', left: '75%', delay: 3 },
          { top: '65%', left: '10%', delay: 4.5 },
        ].map((node, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full"
            style={{ 
              top: node.top, 
              left: node.left,
              backgroundColor: '#FF0066',
              boxShadow: '0 0 15px rgba(255, 0, 102, 0.5)'
            }}
            animate={{ 
              width: [3, 5, 3],
              height: [3, 5, 3],
              opacity: [0.3, 1, 0.3],
              filter: ['blur(1px)', 'blur(2px)', 'blur(1px)']
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity, 
              delay: node.delay,
              ease: "easeInOut" 
            }}
          />
        ))}

        {/* Protective Pulse Ring */}
        <motion.div 
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[85%] h-[85%] border-[0.5px] border-[#FF0066]/10 rounded-full"
          animate={{ 
            scale: [1, 1.05, 1],
            opacity: [0.1, 0.2, 0.1]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>
      
      {/* Premium Finish Overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-white/20 to-transparent pointer-events-none" />
    </div>
  );
}
