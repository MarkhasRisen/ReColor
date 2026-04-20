import { addDoc, collection, serverTimestamp } from 'firebase/firestore';
import { auth, db } from '../../firebaseConfig';

export { auth, db };

// Save a completed 38-plate test session.
// Dual-writes to user private history AND anonymised research collection.
export async function saveTestSession({
  userId,
  score,
  maxScore,
  diagnosis,
  diagnosisCode,
  severity,
  total,
  shuffledOrder,
}) {
  const timestamp = serverTimestamp();

  const privatePayload = {
    score,
    maxScore,
    total,
    diagnosis,
    severity,
    shuffledOrder,
    date: timestamp,
  };

  const researchPayload = {
    diagnosis,
    diagnosisCode,
    severity,
    score,
    maxScore,
    total,
    device: 'Mobile_Client',
    timestamp,
  };

  try {
    await Promise.all([
      addDoc(collection(db, 'users', userId, 'history'), privatePayload),
      addDoc(collection(db, 'research_data_anonymized'), researchPayload),
    ]);
    return true;
  } catch (err) {
    console.warn('[DB] saveTestSession failed:', err);
    return false;
  }
}
