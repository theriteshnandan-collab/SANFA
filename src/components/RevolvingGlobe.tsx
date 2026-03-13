'use client';

import { motion } from 'framer-motion';

export default function RevolvingGlobe() {
  return (
    <div className="relative w-full max-w-lg aspect-square flex items-center justify-center overflow-hidden group">
      {/* The Sphere Shell */}
      <div className="w-[85%] h-[85%] rounded-full relative overflow-hidden bg-white shadow-[0_30px_100px_-20px_rgba(0,0,0,0.1)] transition-all duration-700 group-hover:scale-[1.02] group-hover:shadow-[0_40px_120px_-20px_rgba(255,0,102,0.1)]">
        
        {/* Depth & Shine Overlays */}
        <div className="absolute inset-0 z-10 rounded-full shadow-[inset_-30px_-30px_80px_rgba(0,0,0,0.04),inset_20px_20px_80px_white]" />
        
        {/* God-Tier Map Revolution */}
        <div className="w-full h-full flex absolute top-0 left-0">
          <motion.div 
            className="flex h-full w-[300%] transform-gpu"
            animate={{ x: ["0%", "-66.66%"] }}
            transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
          >
            {/* Detailed Artist Map Segment 1 */}
            <div className="w-1/3 h-full p-8 opacity-[0.9]">
              <img src="/assets/god-globe.png" alt="World Map Texture Segment 1" className="w-full h-full object-contain" />
            </div>
            {/* Detailed Artist Map Segment 2 */}
            <div className="w-1/3 h-full p-8 opacity-[0.9]">
              <img src="/assets/god-globe.png" alt="World Map Texture Segment 2" className="w-full h-full object-contain" />
            </div>
            {/* Seamless Loop Segment 3 */}
            <div className="w-1/3 h-full p-8 opacity-[0.9]">
              <img src="/assets/god-globe.png" alt="World Map Texture Segment 3" className="w-full h-full object-contain" />
            </div>
          </motion.div>
        </div>

        {/* Atmospheric Glow */}
        <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] bg-gradient-to-br from-white to-transparent opacity-80 filter blur-3xl pointer-events-none" />
      </div>

      {/* Kinetic Orbital Protocols (Lemonade Pink) */}
      {[
        { rotateX: 75, rotateY: 15, duration: 25, size: '100%' },
        { rotateX: -60, rotateY: 30, duration: 40, size: '92%', reverse: true },
        { rotateX: 20, rotateY: -70, duration: 60, size: '110%' },
      ].map((orbit, i) => (
        <motion.div 
          key={i}
          className="absolute border-[0.8px] border-[#FF0066]/20 rounded-full pointer-events-none"
          style={{ 
            width: orbit.size, 
            height: orbit.size,
            rotateX: orbit.rotateX, 
            rotateY: orbit.rotateY 
          }}
          animate={{ rotateZ: orbit.reverse ? -360 : 360 }}
          transition={{ duration: orbit.duration, repeat: Infinity, ease: "linear" }}
        />
      ))}

      {/* Floating Spark Nodes */}
      <motion.div
        className="absolute w-3 h-3 bg-[#FF0066] rounded-full filter blur-[1px] shadow-[0_0_10px_#FF0066]"
        animate={{ 
          rotateY: 360,
          scale: [1, 1.2, 1],
          opacity: [0.5, 1, 0.5]
        }}
        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        style={{ transformOrigin: '250px center' }}
      />
    </div>
  );
}
