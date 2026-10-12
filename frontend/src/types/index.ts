// Types mirroring the backend Pydantic schemas (app/schemas.py).

export type ProcessingStage =
  | "queued"
  | "reading"
  | "extracting_text"
  | "detecting_figures"
  | "extracting_images"
  | "ocr"
  | "embedding"
  | "indexing"
  | "ready"
  | "failed";

export type VisualType =
  | "image"
  | "figure"
  | "diagram"
  | "chart"
  | "table"
  | "schematic";

export type SourceKind = "text" | "visual" | "region";

export interface DocumentInfo {
  document_id: string;
  filename: string;
  upload_time: string;
  page_count: number;
  text_chunk_count: number;
  figure_count: number;
  image_count: number;
  region_count: number;
  stage: ProcessingStage;
  progress: number;
  stage_message: string;
  error?: string | null;
}

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Citation {
  kind: SourceKind;
  source_id: string;
  page_number: number;
  label: string;
  score: number;
  snippet: string;
  image_url: string;
  figure_label: string;
}

export interface RetrievedText {
  chunk_id: string;
  page_number: number;
  text: string;
  score: number;
  document_id?: string;
  lexical_rank?: number | null;
  dense_rank?: number | null;
}

export interface ClaimVerification {
  claim: string;
  support: number;
  best_source?: string | null;
  missing_numbers: string[];
  bad_page_refs: number[];
  supported: boolean;
}

export interface VerificationReport {
  status: "ok" | "flagged" | "no_claims" | "refused";
  support_ratio: number;
  claims: ClaimVerification[];
  unsupported: string[];
}

export interface ClauseOut {
  category: string;
  text: string;
  page_number: number;
  chunk_id: string;
  confidence_margin: number;
  risk_level: "low" | "medium" | "high";
  risk_reason: string;
  suggested_edit: string;
  red_flags: string[];
}

export interface ClauseReport {
  document_id: string;
  clauses: ClauseOut[];
  summary: { high: number; medium: number; low: number };
  disclaimer: string;
  warning?: string | null;
}

export interface SummarySentence {
  text: string;
  page: number;
  chunk_id: string;
}

export interface DocumentSummary {
  issue: SummarySentence[];
  headnotes: SummarySentence[];
  facts: SummarySentence[];
  holding: SummarySentence[];
  statutes_cited: string[];
  cases_cited: string[];
  note: string;
}

export interface TimelineEvent {
  date: string;
  text: string;
  page_number: number;
  chunk_id: string;
}

export interface RetrievedVisual {
  visual_id: string;
  page_number: number;
  image_url: string;
  image_type: VisualType;
  figure_label: string;
  ocr_preview: string;
  score: number;
  bounding_box?: BoundingBox | null;
}

export interface QueryResponse {
  question: string;
  answer: string;
  citations: Citation[];
  text_evidence: RetrievedText[];
  visual_evidence: RetrievedVisual[];
  combined_confidence: number;
  provider: string;
  warning: string;
  verification?: VerificationReport | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
  pending?: boolean;
}
