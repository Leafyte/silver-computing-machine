import React, { useState } from 'react';
import { Pressable, SafeAreaView, StatusBar, StyleSheet, Text, TextInput, View } from 'react-native';
import { driverClient } from '../api/driverClient';

interface DriverLoginScreenProps {
  backendConnected: boolean;
  onLogin: (username: string, password: string) => Promise<void>;
}

export const DriverLoginScreen: React.FC<DriverLoginScreenProps> = ({ backendConnected, onLogin }) => {
  const [username, setUsername] = useState('driver');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    setSubmitting(true);
    setError('');
    try {
      await onLogin(username.trim(), password);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not sign in to the driver account.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.card}>
        <View style={styles.logo}><Text style={styles.logoText}>arQ</Text></View>
        <Text style={styles.eyebrow}>DRIVER APP · DEMO</Text>
        <Text style={styles.title}>Sign in to your route</Text>
        <Text style={styles.subtitle}>Your assigned manifest, stop reports and messages sync with dispatch.</Text>
        <View style={[styles.connection, backendConnected ? styles.connected : styles.disconnected]} accessibilityRole="text">
          <View style={[styles.dot, backendConnected ? styles.dotOn : styles.dotOff]} />
          <View style={styles.connectionCopy}>
            <Text style={styles.connectionTitle}>{backendConnected ? 'Backend connected' : 'Can’t reach the backend'}</Text>
            <Text style={styles.connectionText}>{driverClient.getApiBaseUrl()}</Text>
            {!backendConnected && <Text style={styles.connectionHint}>Put your phone and computer on the same Wi-Fi, confirm the backend is running, then restart Expo.</Text>}
          </View>
        </View>
        <TextInput value={username} onChangeText={setUsername} placeholder="Username" autoCapitalize="none" autoCorrect={false} accessibilityLabel="Driver demo username" style={styles.input} />
        <TextInput value={password} onChangeText={setPassword} placeholder="Password" autoCapitalize="none" secureTextEntry accessibilityLabel="Driver demo password" style={styles.input} onSubmitEditing={() => void submit()} />
        {error ? <Text accessibilityRole="alert" style={styles.error}>{error}</Text> : null}
        <Pressable accessibilityRole="button" style={({ pressed }) => [styles.button, pressed && styles.buttonPressed, (!backendConnected || submitting || !password) && styles.buttonDisabled]} onPress={() => void submit()} disabled={!backendConnected || submitting || !username.trim() || !password}>
          <Text style={styles.buttonText}>{submitting ? 'Signing in…' : 'Sign in'}</Text>
        </Pressable>
        <View style={styles.demoBox}>
          <Text style={styles.demoTitle}>Demo credentials</Text>
          <Text style={styles.demoText}>Username: <Text style={styles.demoStrong}>driver</Text></Text>
          <Text style={styles.demoText}>Password: <Text style={styles.demoStrong}>driver-demo-2026</Text></Text>
        </View>
        <Text style={styles.footer}>Demo access only · not for production use</Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F6F8F5', justifyContent: 'center', padding: 22 },
  card: { backgroundColor: '#FFFFFF', borderRadius: 24, padding: 22, borderWidth: 1, borderColor: 'rgba(21, 33, 29, 0.08)' },
  logo: { width: 50, height: 50, borderRadius: 16, backgroundColor: '#E8F5EE', alignItems: 'center', justifyContent: 'center', marginBottom: 20 },
  logoText: { color: '#1E5A44', fontSize: 19, fontWeight: '900' },
  eyebrow: { color: '#1E5A44', fontSize: 10, fontWeight: '800', letterSpacing: 1.5 },
  title: { color: '#15211D', fontSize: 25, fontWeight: '900', marginTop: 8 },
  subtitle: { color: '#68756F', fontSize: 13, lineHeight: 19, marginTop: 8, marginBottom: 18 },
  connection: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: 11, borderRadius: 12, marginBottom: 16 },
  connectionCopy: { flex: 1, gap: 2 },
  connected: { backgroundColor: '#E8F5EE' },
  disconnected: { backgroundColor: '#FEF3C7' },
  dot: { width: 8, height: 8, borderRadius: 4 },
  dotOn: { backgroundColor: '#1E5A44' },
  dotOff: { backgroundColor: '#D78A45' },
  connectionTitle: { color: '#4D5C54', fontSize: 12, fontWeight: '800' },
  connectionText: { color: '#4D5C54', fontSize: 10, fontWeight: '700' },
  connectionHint: { color: '#735018', fontSize: 10, lineHeight: 14, marginTop: 2 },
  input: { backgroundColor: '#FAFAF8', borderWidth: 1, borderColor: '#E5E0D8', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 13, color: '#15211D', fontSize: 14, marginBottom: 12 },
  error: { color: '#C6534C', fontSize: 12, marginBottom: 12 },
  button: { backgroundColor: '#141F1A', paddingVertical: 14, borderRadius: 12, alignItems: 'center', marginTop: 2 },
  buttonDisabled: { opacity: 0.5 },
  buttonPressed: { opacity: 0.88, transform: [{ scale: 0.985 }] },
  buttonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '800' },
  demoBox: { marginTop: 18, backgroundColor: '#F8F7F3', borderRadius: 14, borderWidth: 1, borderColor: '#E5E0D8', padding: 14, gap: 4 },
  demoTitle: { color: '#15211D', fontSize: 11, fontWeight: '800', marginBottom: 3 },
  demoText: { color: '#68756F', fontSize: 11 },
  demoStrong: { color: '#15211D', fontWeight: '800' },
  footer: { color: '#929E99', fontSize: 10, textAlign: 'center', marginTop: 16 },
});
