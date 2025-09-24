-- Supabase / Postgres schema for IA Mercantil MVP
create extension if not exists "uuid-ossp";
create extension if not exists vector;

create table if not exists cases (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz default now(),
  meta jsonb
);

-- Ajusta el tamaño del vector según tu modelo de embeddings (1536 para text-embedding-3-small)
create table if not exists chunks (
  id uuid primary key default uuid_generate_v4(),
  case_id uuid references cases(id) on delete cascade,
  section_id text,
  section_title text,
  para_index int,
  ref text,
  text text,
  embedding vector(1536),
  tsv tsvector
);

-- Indexes
create index if not exists idx_chunks_embedding on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_chunks_tsv on chunks using gin(tsv);
