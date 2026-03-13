'use client';

import { motion } from 'framer-motion';

export default function RevolvingGlobe() {
  return (
    <div className="relative w-full max-w-lg aspect-square flex items-center justify-center overflow-hidden group">
      {/* The Sphere (Circular Mask) */}
      <div className="w-[85%] h-[85%] rounded-full relative overflow-hidden bg-white border border-gray-100 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)] transition-transform duration-700 group-hover:scale-[1.02]">
        
        {/* Shading/Depth Overlay (Gives it a 3D feel) */}
        <div className="absolute inset-0 z-10 rounded-full shadow-[inset_-30px_-30px_70px_rgba(0,0,0,0.03),inset_30px_30px_70px_rgba(255,255,255,1)] pointer-events-none" />
        <div className="absolute inset-0 z-20 rounded-full border-[0.5px] border-black/5 pointer-events-none" />
        
        {/* Seamless Scrolling Map */}
        <motion.div 
          className="flex h-full w-[300%] absolute top-0 left-0"
          animate={{ x: ["0%", "-66.66%"] }}
          transition={{ duration: 45, repeat: Infinity, ease: "linear" }}
        >
          {/* Map Segment 1 */}
          <div className="w-1/3 h-full py-12 px-4 opacity-[0.25] transform-gpu">
            <DetailedWorldMapSVG />
          </div>
          {/* Map Segment 2 */}
          <div className="w-1/3 h-full py-12 px-4 opacity-[0.25] transform-gpu">
            <DetailedWorldMapSVG />
          </div>
          {/* Map Segment 3 (Duplicate for Seamless Loop) */}
          <div className="w-1/3 h-full py-12 px-4 opacity-[0.25] transform-gpu">
            <DetailedWorldMapSVG />
          </div>
        </motion.div>

        {/* Atmosphere Highlight */}
        <div className="absolute top-[5%] left-[15%] w-[40%] h-[30%] bg-gradient-to-br from-white to-transparent opacity-60 rounded-full filter blur-2xl pointer-events-none" />
      </div>

      {/* Orbiting Multi-Axis Rings (Lemonade Style) */}
      <motion.div 
        className="absolute w-full h-full border-[1.2px] border-[#FF0066]/30 rounded-full flex items-center justify-center"
        style={{ rotateX: 70, rotateY: 10 }}
        animate={{ rotateZ: 360 }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      >
        <div className="w-3 h-3 bg-[#FF0066] rounded-full blur-[2px]" />
      </motion.div>

      <motion.div 
        className="absolute w-[92%] h-[92%] border-[0.8px] border-[#FF0066]/20 rounded-full"
        style={{ rotateX: -45, rotateY: 30 }}
        animate={{ rotateZ: -360 }}
        transition={{ duration: 35, repeat: Infinity, ease: "linear" }}
      />

      <motion.div 
        className="absolute w-[110%] h-[110%] border-[0.5px] border-[#FF0066]/10 rounded-full"
        style={{ rotateX: 20, rotateY: -60 }}
        animate={{ rotateZ: 360 }}
        transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

function DetailedWorldMapSVG() {
  return (
    <svg viewBox="0 0 1000 500" className="w-full h-full fill-none stroke-current text-black stroke-[0.8px]">
      {/* North America */}
      <path d="M100,120 Q120,80 160,100 T220,110 Q250,140 230,180 T180,220 Q150,250 110,230 T80,180 Z" />
      {/* South America */}
      <path d="M180,240 Q210,240 230,280 T240,350 Q220,420 180,450 T130,400 Q110,350 140,280 Z" />
      {/* Eurasia */}
      <path d="M450,100 Q550,60 650,80 T750,100 Q850,120 880,180 T800,250 Q750,280 650,260 T550,240 Q450,220 420,160 Z" />
      {/* Africa */}
      <path d="M420,240 Q480,230 530,260 T560,340 Q540,400 480,430 T400,380 Q370,330 400,270 Z" />
      {/* Australia */}
      <path d="M750,320 Q780,300 820,320 T840,380 Q810,420 760,400 T730,350 Z" />
      {/* Islands / Details */}
      <circle cx="280" cy="120" r="5" />
      <circle cx="600" cy="350" r="8" />
      <circle cx="350" cy="400" r="4" />
      <path d="M10,250 L990,250" className="stroke-gray-100 opacity-50" />
      {/* Quirky details */}
      <path d="M480,150 L500,150" className="stroke-[#FF0066]/40" />
      <path d="M700,200 L710,210" className="stroke-[#FF0066]/40" />
    </svg>
  );
}
