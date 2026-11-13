-- Supabase Initial Schema for Karaoke Agent
-- Mirrors Aniden.ai pattern with 500 initial credits, 100 credits per song, and song caching

-- 1. Profiles Table (User Credits & Auth linkage)
create table if not exists public.profiles (
  id uuid primary key references auth.users on delete cascade,
  email text,
  credits int default 500, -- 500 free credits on sign up
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Auto-create profile with 500 credits when user signs up
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, email, credits)
  values (new.id, new.email, 500);
  return new;
end;
$$ language plpgsql security definer;

create or replace trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- 2. Songs Table (Tracks jobs, stems, lyrics, and final video)
create table if not exists public.songs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users not null,
  source_url text,
  source_path text,                 -- Supabase storage path: source-audio bucket
  source_hash text,                 -- SHA256 or URL ID for cache deduplication
  title text,
  artist text,
  language text default 'auto',
  dialect text,
  duration_s numeric,
  official_lyrics text,             -- Optional confirmed/pasted lyrics
  status text not null default 'pending' check (status in (
    'pending',
    'downloading',
    'separating',
    'transcribing',
    'transliterating',
    'lyrics_ready',
    'lyrics_approved',
    'rendering',
    'completed',
    'failed'
  )),
  progress int default 0,
  error_message text,
  instrumental_path text,           -- Supabase storage path: stems bucket
  vocals_path text,                 -- Supabase storage path: stems bucket
  lyrics_json jsonb,                -- Structured lyrics with word-level timestamps
  video_path text,                  -- Supabase storage path: karaoke-videos bucket
  credits_charged int default 100,  -- 100 credits per song
  credits_refunded boolean default false,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- 3. Song Cache Table (Deduplication across users)
create table if not exists public.song_cache (
  source_hash text primary key,
  instrumental_path text,
  vocals_path text,
  lyrics_json jsonb,
  video_path text,
  language text,
  dialect text,
  created_at timestamptz default now()
);

-- 4. User Feedback Table
create table if not exists public.user_feedback (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users,
  song_id uuid references public.songs,
  rating int check (rating between 1 and 5),
  sync_rating int check (sync_rating between 1 and 5),
  phonetic_rating int check (phonetic_rating between 1 and 5),
  comments text,
  created_at timestamptz default now()
);

-- 5. Row Level Security (RLS)
alter table public.profiles enable row level security;
alter table public.songs enable row level security;
alter table public.song_cache enable row level security;
alter table public.user_feedback enable row level security;

-- Profiles: users read their own profile
create policy "Users can read own profile" on public.profiles
  for select using (auth.uid() = id);

-- Songs: users read & update their own songs
create policy "Users can read own songs" on public.songs
  for select using (auth.uid() = user_id);

create policy "Users can insert own songs" on public.songs
  for insert with check (auth.uid() = user_id);

create policy "Users can update own songs" on public.songs
  for update using (auth.uid() = user_id);

-- Song Cache: readable by authenticated users, writable only by service role
create policy "Authenticated users can read cache" on public.song_cache
  for select using (auth.role() = 'authenticated');
