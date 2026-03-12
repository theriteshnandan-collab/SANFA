// Lemon Squeezy Webhook Handler (Brick 4)
// This handles subscription updates and payment success to grant user credits.

import { NextRequest, NextResponse } from 'next/server';
import { supabase } from '@/lib/supabase';
import crypto from 'crypto';

const LE_SIGNING_SECRET = process.env.LEMON_SQUEEZY_WEBHOOK_SECRET!;

export async function POST(req: NextRequest) {
    const rawBody = await req.text();
    
    // 1. Verify Signature
    const hmac = crypto.createHmac('sha256', LE_SIGNING_SECRET);
    const digest = Buffer.from(hmac.update(rawBody).digest('hex'), 'utf8');
    const signature = Buffer.from(req.headers.get('x-signature') || '', 'utf8');

    if (signature.length !== digest.length || !crypto.timingSafeEqual(digest, signature)) {
        return NextResponse.json({ error: 'Invalid signature' }, { status: 401 });
    }

    const payload = JSON.parse(rawBody);
    const eventName = payload['meta']['event_name'];
    const customData = payload['meta']['custom_data'];
    const userId = customData?.user_id;

    if (!userId) {
        return NextResponse.json({ error: 'No user_id found in metadata' }, { status: 400 });
    }

    // 2. Route Events
    if (eventName === 'order_created') {
        // Order successful -> Grant credits / Update subscription
        const productId = payload['data']['attributes']['first_order_item']['product_id'];
        
        // Map product IDs to roles
        let quota = 5;
        let role = 'free';
        if (productId === 'PRO_PLAN_ID') { quota = 100; role = 'pro'; }
        if (productId === 'GOD_MODE_ID') { quota = 9999; role = 'god_mode'; }

        await supabase
            .from('profiles')
            .update({ 
                subscription_status: role,
                credits_remaining: quota,
                subscription_id: payload['data']['id']
            })
            .eq('id', userId);
    }

    return NextResponse.json({ received: true });
}
