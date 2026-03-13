'use client';

import { motion } from 'framer-motion';

export default function RevolvingGlobe() {
  return (
    <div className="relative w-full max-w-md aspect-square flex items-center justify-center overflow-hidden">
      {/* The Sphere (Circular Mask) */}
      <div className="w-[80%] h-[80%] rounded-full relative overflow-hidden bg-white border border-gray-100 shadow-2xl">
        
        {/* Shading/Depth Overlay (Gives it a 3D feel) */}
        <div className="absolute inset-0 z-10 rounded-full shadow-[inset_-20px_-20px_50px_rgba(0,0,0,0.02),inset_20px_20px_50px_rgba(255,255,255,0.8)] pointer-events-none" />
        
        {/* Seamless Scrolling Map */}
        <motion.div 
          className="flex h-full w-[200%] absolute top-0 left-0"
          animate={{ x: ["0%", "-50%"] }}
          transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
        >
          {/* Map Segment 1 */}
          <div className="w-1/2 h-full py-10 opacity-20 transform-gpu">
            <WorldMapSVG />
          </div>
          {/* Map Segment 2 (Duplicate for Seamless Loop) */}
          <div className="w-1/2 h-full py-10 opacity-20 transform-gpu">
            <WorldMapSVG />
          </div>
        </motion.div>

        {/* Glossy Reflection */}
        <div className="absolute top-[10%] left-[20%] w-[30%] h-[20%] bg-white opacity-40 rounded-full filter blur-xl pointer-events-none" />
      </div>

      {/* Orbiting Rings (Quirky Fine-Line style) */}
      <motion.div 
        className="absolute w-full h-full border-[1.5px] border-[#FF0066]/30 rounded-full"
        style={{ rotateX: 65, rotateY: 15 }}
        animate={{ rotateZ: 360 }}
        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
      />
      <motion.div 
        className="absolute w-[95%] h-[95%] border-[1.5px] border-[#FF0066]/20 rounded-full"
        style={{ rotateX: 45, rotateY: -25 }}
        animate={{ rotateZ: -360 }}
        transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

function WorldMapSVG() {
  return (
    <svg viewBox="0 0 1000 500" className="w-full h-full fill-none stroke-current text-black stroke-[0.5px]">
      {/* Simple stylized world map paths */}
      <path d="M150,150 Q200,100 250,150 T350,150 Q400,200 350,250 T250,250 Q200,300 150,250 T150,150" />
      <path d="M550,200 Q600,150 650,200 T750,200 Q800,250 750,300 T650,300 Q600,350 550,300 T550,200" />
      <path d="M400,350 Q450,300 500,350 T600,350 Q650,400 600,450 T500,450 Q450,500 400,450 T400,350" />
      <circle cx="100" cy="400" r="10" />
      <circle cx="800" cy="100" r="15" />
      <path d="M0,250 L1000,250" className="stroke-gray-100" />
    </svg>
  );
}
