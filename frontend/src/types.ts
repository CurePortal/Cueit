export type Section = {
  start: number;
  end: number;
  energy: number;
  kind: string;
};

export type AnalysisResult = {
  path: string;
  duration: number;
  bpm: number;
  bpm_confidence: number;
  key: string;
  scale: string;
  key_strength: number;
  camelot: string;
  beats: number[];
  downbeats: number[];
  onsets: number[];
  sections: Section[];
  rms_times: number[];
  rms: number[];
  peak_mins: number[];
  peak_maxs: number[];
  integrated_loudness: number;
  beat_energy: number[];
  beat_low: number[];
  beat_flux: number[];
  downbeat_phase: number;
  downbeat_confidence: number;
  phrase_length: number;
};

export type Cue = {
  name: string;
  time: number;
  color: string;
  kind: string;
};

export type TrackData = {
  path: string;
  analysis: AnalysisResult;
  cues: Cue[];
};

export type JobDto = {
  path: string;
  status: string;
  error: string;
  track: TrackData | null;
};

export type LibDto = {
  name: string;
  kind: string;
  locator: string;
  tracks: number | null;
};

export type Progress = {
  done: number;
  total: number;
  path: string;
  status: string;
  error: string;
  track: TrackData | null;
};
