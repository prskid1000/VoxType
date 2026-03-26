import http from 'http';
import https from 'https';

// Generate a minimal valid WAV file (0.1s silence, 16kHz mono 16-bit)
function silentWav(): Buffer {
  const sampleRate = 16000;
  const numSamples = sampleRate / 10; // 0.1s
  const dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
  const buf = Buffer.alloc(44 + dataSize, 0); // header + silent PCM
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16); // fmt chunk size
  buf.writeUInt16LE(1, 20);  // PCM
  buf.writeUInt16LE(1, 22);  // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buf.writeUInt16LE(2, 32);  // block align
  buf.writeUInt16LE(16, 34); // bits per sample
  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);
  return buf;
}

export async function preloadWhisper(whisperUrl: string): Promise<void> {
  console.log('[VoxType] Preloading Whisper model...');
  try {
    await transcribe(silentWav(), whisperUrl);
    console.log('[VoxType] Whisper model preloaded');
  } catch (e: any) {
    console.log(`[VoxType] Whisper preload failed (non-fatal): ${e.message}`);
  }
}

export async function transcribe(audioBuffer: Buffer, whisperUrl: string): Promise<string> {
  const url = new URL('/v1/audio/transcriptions', whisperUrl);

  const boundary = '----VoxTypeBoundary' + Date.now();
  const parts: Buffer[] = [];

  // file field
  const fileHeader = [
    `--${boundary}`,
    'Content-Disposition: form-data; name="file"; filename="audio.webm"',
    'Content-Type: audio/webm',
    '',
    '',
  ].join('\r\n');
  parts.push(Buffer.from(fileHeader, 'utf-8'));
  parts.push(audioBuffer);

  // model field
  const modelField = [
    '',
    `--${boundary}`,
    'Content-Disposition: form-data; name="model"',
    '',
    'whisper-1',
  ].join('\r\n');
  parts.push(Buffer.from(modelField, 'utf-8'));

  // language field
  const langField = [
    '',
    `--${boundary}`,
    'Content-Disposition: form-data; name="language"',
    '',
    'en',
  ].join('\r\n');
  parts.push(Buffer.from(langField, 'utf-8'));

  // response_format field
  const formatField = [
    '',
    `--${boundary}`,
    'Content-Disposition: form-data; name="response_format"',
    '',
    'json',
  ].join('\r\n');
  parts.push(Buffer.from(formatField, 'utf-8'));

  // closing boundary
  parts.push(Buffer.from(`\r\n--${boundary}--\r\n`, 'utf-8'));

  const body = Buffer.concat(parts);

  return new Promise((resolve, reject) => {
    const transport = url.protocol === 'https:' ? https : http;
    const req = transport.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length,
      },
    }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const raw = Buffer.concat(chunks).toString('utf-8');
        if (res.statusCode !== 200) {
          reject(new Error(`Whisper STT error ${res.statusCode}: ${raw}`));
          return;
        }
        try {
          const json = JSON.parse(raw);
          resolve(json.text || '');
        } catch {
          // If not JSON, return raw text
          resolve(raw.trim());
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}
