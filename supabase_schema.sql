```
-- SANFA Cloud Infrastructure Schema (Brick 4)
-- This schema handles user subscriptions, image processing logs, and credits.

-- 1. Profiles table (Extends Supabase Auth)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    subscription_status TEXT DEFAULT 'free', -- 'free', 'pro', 'god_mode'
    subscription_id TEXT, -- Lemon Squeezy Sub ID
    credits_remaining INTEGER DEFAULT 5, -- Free users get 5 processed images
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indices for 200k/mo performance
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);
CREATE INDEX IF NOT EXISTS idx_profiles_subscription ON public.profiles(subscription_status);

-- 2. Protected Images Log
CREATE TABLE IF NOT EXISTS public.protected_images (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    original_filename TEXT NOT NULL,
    original_hash TEXT NOT NULL, -- SHA256 to prevent duplicate processing
    storage_path TEXT NOT NULL, -- Path in Supabase Storage
    engine_version TEXT DEFAULT 'V5',
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Performance indices for high-volume protection jobs
CREATE INDEX IF NOT EXISTS idx_protected_images_user_id ON public.protected_images(user_id);
CREATE INDEX IF NOT EXISTS idx_protected_images_status ON public.protected_images(status);
CREATE INDEX IF NOT EXISTS idx_protected_images_created_at ON public.protected_images(processed_at DESC);

-- 3. Row Level Security (RLS)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.protected_images ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only see/edit their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- Images: Users can only see their own images
CREATE POLICY "Users can view own images" ON public.protected_images
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own images" ON public.protected_images
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- 4. Triggers for updated_at
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER on_profiles_updated
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();
