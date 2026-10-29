import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  SafeAreaView,
  TextInput
} from 'react-native';
import { DataBadge } from '../components/DataBadge';
import { SimulatedActionNotice } from '../components/SimulatedActionNotice';
import { DRIVER_PROFILE } from '../data/mockDriverData';
import { DriverIdentity } from '../api/driverClient';

interface ProfileScreenProps {
  identity: DriverIdentity | null;
  onPair: (code: string) => Promise<void>;
  onLogout: () => Promise<void>;
}

export const ProfileScreen: React.FC<ProfileScreenProps> = ({ identity, onPair, onLogout }) => {
  const [offlineSync, setOfflineSync] = useState<boolean>(true);
  const [hapticFeedback, setHapticFeedback] = useState<boolean>(true);
  const [actionNotice, setActionNotice] = useState<{ action: string; detail: string } | null>(null);
  const [pairingCode, setPairingCode] = useState('');
  const [isPairing, setIsPairing] = useState(false);

  const handlePair = async () => {
    setIsPairing(true);
    try { await onPair(pairingCode.trim()); setPairingCode(''); }
    catch (error) { setActionNotice({ action: 'Driver pairing failed', detail: error instanceof Error ? error.message : 'Could not pair this device.' }); }
    finally { setIsPairing(false); }
  };

  const handleToggleOfflineSync = (val: boolean) => {
    setOfflineSync(val);
    setActionNotice({
      action: `Offline Manifest Cache ${val ? 'Enabled' : 'Disabled'}`,
      detail: `Stop waypoints and consignment payloads ${
        val ? 'cached in device local memory' : 'purged from local storage'
      }. Local preview configuration only.`
    });
  };

  const handleToggleHaptics = (val: boolean) => {
    setHapticFeedback(val);
    setActionNotice({
      action: `Tactile Feedback ${val ? 'Enabled' : 'Disabled'}`,
      detail: `Corridor alert vibration threshold set to ${val ? 'Active' : 'Off'}. No sensor permission required.`
    });
  };

  const handleCallDispatch = () => {
    setActionNotice({
      action: 'Contacting Central Dispatch Desk',
      detail: 'Simulated telephony trigger to Bangalore Hub Operations (+91 80 4910 8800). No native cellular permission requested.'
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>04 · DRIVER DOSSIER</Text>
            <Text style={styles.title}>Operator Profile</Text>
          </View>
          <DataBadge label={identity ? 'Backend session' : 'Preview'} tone={identity ? 'success' : 'preview'} />
        </View>

        <View style={styles.sectionCard}>
          <Text style={styles.sectionHeading}>{identity ? 'Signed-in Driver Session' : 'Pair this Driver Device'}</Text>
          {identity ? <>
            <Text style={styles.rowValueBold}>Driver {identity.driver_id}</Text>
            <Text style={styles.rowValue}>Assigned vehicle {identity.vehicle_id}</Text>
            <Text style={styles.rowValue}>Session expires {identity.expires_at}</Text>
            <TouchableOpacity style={styles.dispatchCallBtn} onPress={onLogout}><Text style={styles.dispatchCallBtnText}>Sign out</Text></TouchableOpacity>
          </> : <>
            <Text style={styles.rowValue}>Enter the one-time pairing code issued for your driver and vehicle.</Text>
            <TextInput value={pairingCode} onChangeText={setPairingCode} placeholder="One-time pairing code" autoCapitalize="none" secureTextEntry style={styles.pairingInput} accessibilityLabel="One-time driver pairing code" />
            <TouchableOpacity style={styles.dispatchCallBtn} onPress={handlePair} disabled={isPairing || !pairingCode.trim()}><Text style={styles.dispatchCallBtnText}>{isPairing ? 'Pairing…' : 'Pair with backend'}</Text></TouchableOpacity>
          </>}
        </View>

        {/* Global Action Feedback Notice */}
        {actionNotice && (
          <SimulatedActionNotice
            actionName={actionNotice.action}
            detail={actionNotice.detail}
            onDismiss={() => setActionNotice(null)}
          />
        )}

        {/* Profile Identity Card (Vanta Ride Spacing) */}
        {!identity && <View style={styles.profileCard}>
          <View style={styles.avatarRow}>
            <View style={styles.avatar}>
              <Text style={styles.avatarText}>RG</Text>
            </View>
            <View style={styles.profileInfo}>
              <Text style={styles.driverName}>{DRIVER_PROFILE.name}</Text>
              <Text style={styles.driverId}>{DRIVER_PROFILE.driverId} · {DRIVER_PROFILE.rating} ★ ({DRIVER_PROFILE.completedTrips} trips)</Text>
              <Text style={styles.driverPhone}>{DRIVER_PROFILE.phone}</Text>
            </View>
          </View>

          {/* Quick Contact Dispatch CTA */}
          <TouchableOpacity
            style={styles.dispatchCallBtn}
            onPress={handleCallDispatch}
            activeOpacity={0.7}
            hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}
          >
            <Text style={styles.dispatchCallBtnText}>Call Central Dispatch Desk (Simulated)</Text>
          </TouchableOpacity>
        </View>}

        {/* Assigned Vehicle Dossier */}
        {!identity && <View style={styles.sectionCard}>
          <Text style={styles.sectionHeading}>Assigned Electric Fleet Unit</Text>

          <View style={styles.row}>
            <Text style={styles.rowLabel}>Callsign</Text>
            <Text style={styles.rowValueBold}>{DRIVER_PROFILE.vehicleCallsign}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Vehicle Model</Text>
            <Text style={styles.rowValue}>{DRIVER_PROFILE.vehicleModel}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Registration Number</Text>
            <Text style={styles.rowValueMono}>{DRIVER_PROFILE.licenseNumber}</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Payload Capacity</Text>
            <Text style={styles.rowValue}>{DRIVER_PROFILE.maxPayloadKg} kg Max Payload</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Battery State of Charge</Text>
            <Text style={[styles.rowValueBold, { color: '#205A46' }]}>{DRIVER_PROFILE.batteryPct}% Nominal</Text>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Shift Allocation</Text>
            <Text style={styles.rowValue}>{DRIVER_PROFILE.shiftId} ({DRIVER_PROFILE.shiftHours})</Text>
          </View>
        </View>}

        {/* Honest System Governance Notice (AGENTS.md §7 & §8) */}
        <View style={styles.noticeCard}>
          <Text style={styles.noticeHeading}>DATA SOURCES & CONNECTIONS</Text>
          <Text style={styles.noticeBody}>
            {identity ? 'Manifest and stop/delay reports sync to the backend. Road navigation, GPS, consignee contact, proof of delivery, and shift services are not provided by the current API.' : 'Pair a driver session to load the assigned backend manifest and submit stop or delay reports. Without a session, route screens use preview fixtures.'}
          </Text>
          <View style={styles.noticePills}>
            {!identity && <Text style={styles.noticePill}>₹83/$ assumed rate</Text>}
            <Text style={styles.noticePill}>Simulated ETAs</Text>
            <Text style={styles.noticePill}>No speed-up claimed</Text>
          </View>
        </View>

        {/* Device & Sync Settings */}
        <View style={styles.sectionCard}>
          <Text style={styles.sectionHeading}>Terminal Configuration</Text>

          <View style={styles.switchRow}>
            <View style={styles.switchTextWrap}>
              <Text style={styles.switchTitle}>Offline Manifest Cache</Text>
              <Text style={styles.switchSubtitle}>Local storage of stop waypoints and consignment payloads</Text>
            </View>
            <Switch
              value={offlineSync}
              onValueChange={handleToggleOfflineSync}
              trackColor={{ false: '#EEF2ED', true: '#205A46' }}
            />
          </View>

          <View style={styles.switchRow}>
            <View style={styles.switchTextWrap}>
              <Text style={styles.switchTitle}>Haptic Feedback</Text>
              <Text style={styles.switchSubtitle}>Tactile impulse on corridor delay alert trigger</Text>
            </View>
            <Switch
              value={hapticFeedback}
              onValueChange={handleToggleHaptics}
              trackColor={{ false: '#EEF2ED', true: '#205A46' }}
            />
          </View>
        </View>

        {/* Version Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>arQ Driver HUD Preview · v1.4.0 (Expo Android Native)</Text>
          <Text style={styles.footerText}>QML-01 Dispatch Logistics OS · Bangalore Hub</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F6F8F5',
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 90,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 14,
  },
  eyebrow: {
    fontSize: 10,
    fontWeight: '700',
    color: '#77736D',
    letterSpacing: 0.5,
    fontFamily: 'monospace',
  },
  title: {
    fontSize: 20,
    fontWeight: '800',
    color: '#15211D',
    marginTop: 2,
  },
  profileCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
    marginBottom: 16,
  },
  avatarRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 16,
    backgroundColor: '#17231F',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 14,
  },
  avatarText: {
    color: '#A8DBC9',
    fontSize: 18,
    fontWeight: '800',
  },
  profileInfo: {
    flex: 1,
  },
  driverName: {
    fontSize: 16,
    fontWeight: '800',
    color: '#15211D',
  },
  driverId: {
    fontSize: 11,
    color: '#77736D',
    marginTop: 1,
  },
  driverPhone: {
    fontSize: 11,
    fontWeight: '600',
    color: '#205A46',
    marginTop: 2,
  },
  dispatchCallBtn: {
    marginTop: 14,
    backgroundColor: '#F4F6F4',
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
  },
  dispatchCallBtnText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#15211D',
  },
  sectionCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
    marginBottom: 16,
  },
  pairingInput: {
    backgroundColor: '#F6F8F5',
    borderWidth: 1,
    borderColor: '#DCE4DD',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 11,
    marginTop: 12,
    color: '#15211D',
  },
  sectionHeading: {
    fontSize: 13,
    fontWeight: '800',
    color: '#15211D',
    marginBottom: 12,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(21, 33, 29, 0.04)',
  },
  rowLabel: {
    fontSize: 11,
    color: '#55534F',
  },
  rowValue: {
    fontSize: 11,
    color: '#15211D',
    fontWeight: '600',
  },
  rowValueBold: {
    fontSize: 11,
    color: '#15211D',
    fontWeight: '800',
  },
  rowValueMono: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#15211D',
  },
  noticeCard: {
    backgroundColor: '#F8F6F1',
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: '#E2DEC9',
    borderLeftWidth: 4,
    borderLeftColor: '#205A46',
    marginBottom: 16,
  },
  noticeHeading: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: '800',
    color: '#205A46',
  },
  noticeBody: {
    fontSize: 11,
    color: '#55534F',
    marginTop: 4,
    lineHeight: 16,
  },
  noticePills: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
  },
  noticePill: {
    fontSize: 10,
    fontWeight: '700',
    color: '#4A5565',
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 4,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(21, 33, 29, 0.04)',
  },
  switchTextWrap: {
    flex: 1,
    marginRight: 10,
  },
  switchTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#15211D',
  },
  switchSubtitle: {
    fontSize: 10,
    color: '#77736D',
    marginTop: 1,
  },
  footer: {
    alignItems: 'center',
    paddingVertical: 16,
    gap: 2,
  },
  footerText: {
    fontSize: 10,
    color: '#929E99',
    fontFamily: 'monospace',
  },
});
