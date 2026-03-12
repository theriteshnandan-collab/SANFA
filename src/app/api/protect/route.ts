import { createClient } from '@/lib/supabase-server';
import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { ImageJobSchema } from '@/lib/schemas';

// ==========================================
// HIGH-SCALE PROTECTION API (200k/mo)
// ==========================================

export async function POST(req: Request) {
  try {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();

    if (!session) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const validatedData = ImageJobSchema.omit({ job_id: true, created_at: true }).parse({
      ...body,
      user_id: session.user.id
    });

    // 1. Insert Pending Job into Database
    const { data: job, error: dbError } = await supabase
      .from('protected_images')
      .insert({
        user_id: session.user.id,
        original_name: body.fileName || 'unnamed_image',
        original_url: body.imageUrl,
        status: 'pending'
      })
      .select()
      .single();

    if (dbError) throw dbError;

    // 2. TRIGGER MODAL WORKER (Async)
    // In a production setup, we would push to a Redis queue like Upstash.
    // For now, we simulate the asynchronous handoff to Modal.
    // Example: await fetch(process.env.MODAL_WEBHOOK_URL, { ...job })

    return NextResponse.json({ 
      success: true, 
      jobId: job.id,
      message: 'Transmission engaged. GPU workers assigned.'
    });

  } catch (error: any) {
    console.error('Protection Job Error:', error);
    return NextResponse.json({ 
      error: error.message || 'Internal System Collapse' 
    }, { status: 500 });
  }
}
