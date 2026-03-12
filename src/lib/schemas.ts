import { z } from 'zod';

// ==========================================
// THE CONTRACT: Type Safety for 200k/mo Scale
// ==========================================

export const UserProfileSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  subscription_status: z.enum(['free', 'pro', 'god_mode']),
  credits_remaining: z.number().int().nonnegative(),
  created_at: z.string().datetime(),
});

export const ImageJobSchema = z.object({
  job_id: z.string().uuid(),
  user_id: z.string().uuid(),
  original_url: z.string().url(),
  status: z.enum(['pending', 'processing', 'completed', 'failed']),
  engine_version: z.string().default('V5'),
  clip_epsilon: z.number().optional(),
  created_at: z.string().datetime(),
});

export const ProtectionResultSchema = z.object({
  job_id: z.string().uuid(),
  protected_url: z.string().url(),
  clip_distance: z.number(),
  pixels_modified_pct: z.number(),
  processing_time_ms: z.number(),
  hashes: z.object({
    original: z.string(),
    protected: z.string()
  })
});

export type UserProfile = z.infer<typeof UserProfileSchema>;
export type ImageJob = z.infer<typeof ImageJobSchema>;
export type ProtectionResult = z.infer<typeof ProtectionResultSchema>;
