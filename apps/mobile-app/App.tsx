import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { TodayScreen } from './src/screens/TodayScreen';
import { LiveRouteScreen } from './src/screens/LiveRouteScreen';
import { LoadsScreen } from './src/screens/LoadsScreen';
import { ProfileScreen } from './src/screens/ProfileScreen';
import { BackendDriverState, DriverIdentity, driverClient } from './src/api/driverClient';
import {
  DRIVER_STOPS,
  DOCK_CHECKLIST,
  DriverStop,
  DockChecklistItem
} from './src/data/mockDriverData';

type Tab = 'today' | 'live' | 'loads' | 'profile';
type BackendState = 'checking' | 'connected' | 'offline';

const toPlanClock = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? `T+${value.toFixed(1)} units` : 'Unavailable';

function toDriverStops(state: BackendDriverState): DriverStop[] {
  const latestDelayByStop = new Set(state.reports.filter((report) => report.kind === 'delay_report' && report.stop_id).map((report) => report.stop_id));
  return state.manifest.ordered_stops.map((row, index) => {
    const id = String(row.order_id ?? `stop-${index + 1}`);
    const status = state.stop_status[id] ?? 'pending';
    return {
      id, sequence: index + 1, customerName: id, consignee: 'Consignee details not supplied',
      address: 'Address not supplied in dispatch scenario', trackingId: id, packageCount: 0,
      windowStart: toPlanClock(row.window_start), windowEnd: toPlanClock(row.window_end),
      etaInterval: `${toPlanClock(row.service_start ?? row.arrival)} · scenario estimate`, pLate: 'Unavailable', demandKg: 0,
      status: status === 'completed' ? 'completed' : status === 'arrived' ? 'in_transit' : latestDelayByStop.has(id) ? 'at_risk' : 'pending',
      isLocked: status === 'completed', notes: 'Backend manifest stop. Location and consignee metadata are unavailable.',
      corridorAdvisory: latestDelayByStop.has(id) ? 'Driver delay report sent to dispatch for review.' : undefined,
      contactName: 'Not supplied', contactPhone: '', contactRole: 'Not supplied', specialHandling: 'Not supplied',
      maneuverInstruction: 'Road navigation is not connected', maneuverDistance: 'Unavailable',
      maneuverNext: 'Use an approved navigation app', maneuverIcon: '·',
    };
  });
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('today');
  const [stops, setStops] = useState<DriverStop[]>(DRIVER_STOPS);
  const [currentNavStop, setCurrentNavStop] = useState<DriverStop>(DRIVER_STOPS[1]);
  const [selectedDate, setSelectedDate] = useState<string>('29 Oct');
  const [shiftActive, setShiftActive] = useState<boolean>(true);
  const [checklist, setChecklist] = useState<DockChecklistItem[]>(DOCK_CHECKLIST);
  const [backendState, setBackendState] = useState<BackendState>('checking');
  const [driverIdentity, setDriverIdentity] = useState<DriverIdentity | null>(null);
  const [manifestState, setManifestState] = useState<BackendDriverState | null>(null);
  const [backendMessage, setBackendMessage] = useState<string | null>(null);

  const refreshDriverState = async () => {
    const state = await driverClient.getState();
    setManifestState(state);
    setDriverIdentity(driverClient.getIdentity());
    const assignedStops = toDriverStops(state);
    setStops(assignedStops);
    setCurrentNavStop(assignedStops.find((stop) => stop.status !== 'completed') ?? assignedStops[0] ?? DRIVER_STOPS[0]);
  };

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        if (!driverClient.isPaired()) await driverClient.restoreSession();
        const connected = await driverClient.getHealth();
        if (active) setBackendState(connected ? 'connected' : 'offline');
        if (connected && driverClient.isPaired()) await refreshDriverState();
      } catch (error) {
        if (active) setBackendMessage(error instanceof Error ? error.message : 'Could not refresh driver manifest.');
      }
    };
    void refresh();
    const interval = setInterval(refresh, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const handleNavigateToLiveRoute = (stop: DriverStop) => {
    setCurrentNavStop(stop);
    setActiveTab('live');
  };

  const handleToggleComplete = async (stop: DriverStop): Promise<boolean> => {
    if (driverClient.isPaired() && manifestState) {
      if (stop.status === 'completed') return false;
      try {
        await driverClient.report({ event_id: `driver-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, manifest_version: manifestState.manifest.manifest_version, kind: 'stop_completed', stop_id: stop.id });
        await refreshDriverState();
        return true;
      } catch (error) {
        setBackendMessage(error instanceof Error ? error.message : 'Stop completion was rejected.');
        return false;
      }
    }
    const isNowDone = stop.status !== 'completed';
    setStops(prev =>
      prev.map(s =>
        s.id === stop.id
          ? { ...s, status: isNowDone ? 'completed' : 'pending' }
          : s
      )
    );
    if (currentNavStop.id === stop.id) {
      setCurrentNavStop(prev => ({
        ...prev,
        status: isNowDone ? 'completed' : 'pending'
      }));
    }
    return true;
  };

  const handleStopArrived = async (arrivedStop: DriverStop): Promise<boolean> => {
    if (driverClient.isPaired() && manifestState) {
      try {
        await driverClient.report({ event_id: `driver-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, manifest_version: manifestState.manifest.manifest_version, kind: 'stop_arrived', stop_id: arrivedStop.id });
        await refreshDriverState();
        return true;
      } catch (error) {
        setBackendMessage(error instanceof Error ? error.message : 'Arrival report was rejected.');
        return false;
      }
    }
    setStops(prev => {
      const updated = prev.map(s => s.id === arrivedStop.id ? { ...s, status: 'in_transit' as const } : s);
      setCurrentNavStop(updated.find(s => s.id === arrivedStop.id) ?? arrivedStop);
      return updated;
    });
    return true;
  };

  const handleReportDelay = async (stopId: string, reason: string, minutes: number): Promise<boolean> => {
    if (driverClient.isPaired() && manifestState) {
      try {
        await driverClient.report({ event_id: `driver-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, manifest_version: manifestState.manifest.manifest_version, kind: 'delay_report', stop_id: stopId, severity: minutes >= 20 ? 'heavy' : minutes >= 15 ? 'moderate' : 'light', note: `${reason} · driver estimate ${minutes} minutes` });
        await refreshDriverState();
        return true;
      } catch (error) {
        setBackendMessage(error instanceof Error ? error.message : 'Delay report was rejected.');
        return false;
      }
    }
    setStops(prev =>
      prev.map(s =>
        s.id === stopId
          ? {
              ...s,
              corridorAdvisory: `Reported by Driver: +${minutes} min (${reason})`,
              pLate: `${Math.min(95, parseInt(s.pLate) + minutes * 2)}% (simulated)`,
              status: 'at_risk' as const
            }
          : s
      )
    );
    if (currentNavStop.id === stopId) {
      setCurrentNavStop(prev => ({
        ...prev,
        corridorAdvisory: `Reported by Driver: +${minutes} min (${reason})`,
        pLate: `${Math.min(95, parseInt(prev.pLate) + minutes * 2)}% (simulated)`,
        status: 'at_risk'
      }));
    }
    return true;
  };

  const handleToggleCheck = (id: string) => {
    setChecklist(prev =>
      prev.map(item => (item.id === id ? { ...item, checked: !item.checked } : item))
    );
  };

  const isDarkTab = activeTab === 'live';

  return (
    <View style={[styles.root, isDarkTab ? styles.rootDark : styles.rootLight]}>
      <StatusBar style={isDarkTab ? 'light' : 'dark'} />

      <View style={styles.backendStatus} accessibilityRole="text">
        <View style={[styles.statusDot, backendState === 'connected' ? styles.statusDotConnected : backendState === 'offline' ? styles.statusDotOffline : styles.statusDotChecking]} />
        <Text style={styles.backendStatusText}>
          {backendState === 'connected' ? (driverIdentity ? `Driver session · ${driverIdentity.vehicle_id} · reports sync to backend` : 'Backend connected · pair your driver account to load its manifest') : backendState === 'offline' ? 'Preview mode · backend unavailable' : 'Checking backend…'}
        </Text>
      </View>
      {backendMessage && <TouchableOpacity onPress={() => setBackendMessage(null)} style={styles.backendMessage}><Text style={styles.backendMessageText}>{backendMessage} · tap to dismiss</Text></TouchableOpacity>}

      {/* Screen Container with Persistent Tab Views */}
      <View style={styles.screenContainer}>
        <View style={[styles.tabContent, activeTab === 'today' ? styles.tabContentVisible : styles.tabContentHidden]}>
          <TodayScreen
            stops={stops}
            onToggleComplete={handleToggleComplete}
            onNavigateToLiveRoute={handleNavigateToLiveRoute}
            selectedDate={selectedDate}
            onSelectDate={setSelectedDate}
            shiftActive={shiftActive}
            onToggleShift={() => setShiftActive(prev => !prev)}
            driverIdentity={driverIdentity}
          />
        </View>

        <View style={[styles.tabContent, activeTab === 'live' ? styles.tabContentVisible : styles.tabContentHidden]}>
          <LiveRouteScreen
            activeStop={currentNavStop}
            allStops={stops}
            onSelectStop={setCurrentNavStop}
            onStopArrived={handleStopArrived}
            onReportDelay={handleReportDelay}
            isBackendManifest={Boolean(driverIdentity && manifestState)}
          />
        </View>

        <View style={[styles.tabContent, activeTab === 'loads' ? styles.tabContentVisible : styles.tabContentHidden]}>
          <LoadsScreen
            checklist={checklist}
            onToggleCheck={handleToggleCheck}
            backendManifestActive={Boolean(driverIdentity && manifestState)}
          />
        </View>

        <View style={[styles.tabContent, activeTab === 'profile' ? styles.tabContentVisible : styles.tabContentHidden]}>
          <ProfileScreen
            identity={driverIdentity}
            onPair={async (code) => { await driverClient.pair(code); await refreshDriverState(); }}
            onLogout={async () => { await driverClient.logout(); setDriverIdentity(null); setManifestState(null); setStops(DRIVER_STOPS); setCurrentNavStop(DRIVER_STOPS[1]); }}
          />
        </View>
      </View>

      {/* Custom Bottom Tab Bar (Vanta Ride Spacing & High-Contrast Glyphs) */}
      <View style={[styles.tabBar, isDarkTab ? styles.tabBarDark : styles.tabBarLight]}>
        {[
          { id: 'today', label: 'Today', glyph: '◫' },
          { id: 'live', label: 'Live route', glyph: '◎' },
          { id: 'loads', label: 'Loads', glyph: '▤' },
          { id: 'profile', label: 'Profile', glyph: '○' },
        ].map(tab => {
          const isSelected = activeTab === tab.id;
          return (
            <TouchableOpacity
              key={tab.id}
              style={[
                styles.tabItem,
                isSelected && (isDarkTab ? styles.tabItemActiveDark : styles.tabItemActiveLight),
              ]}
              onPress={() => setActiveTab(tab.id as Tab)}
              activeOpacity={0.7}
              accessibilityRole="tab"
              accessibilityState={{ selected: isSelected }}
              accessibilityLabel={tab.label}
            >
              <Text
                style={[
                  styles.tabGlyph,
                  isDarkTab ? styles.tabGlyphDark : styles.tabGlyphLight,
                  isSelected && (isDarkTab ? styles.tabGlyphActiveDark : styles.tabGlyphActiveLight),
                ]}
              >
                {tab.glyph}
              </Text>
              <Text
                style={[
                  styles.tabLabel,
                  isDarkTab ? styles.tabLabelDark : styles.tabLabelLight,
                  isSelected && (isDarkTab ? styles.tabLabelActiveDark : styles.tabLabelActiveLight),
                ]}
              >
                {tab.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  rootLight: {
    backgroundColor: '#F6F8F5',
  },
  rootDark: {
    backgroundColor: '#0F1713',
  },
  backendStatus: {
    minHeight: 30,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    backgroundColor: '#EAF2EB',
  },
  backendStatusText: {
    color: '#45564C',
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.15,
  },
  backendMessage: { paddingHorizontal: 16, paddingVertical: 6, backgroundColor: '#FEF3C7' },
  backendMessageText: { color: '#8A5100', fontSize: 10, fontWeight: '600' },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  statusDotConnected: { backgroundColor: '#16834A' },
  statusDotOffline: { backgroundColor: '#D97706' },
  statusDotChecking: { backgroundColor: '#718078' },
  screenContainer: {
    flex: 1,
  },
  tabContent: {
    flex: 1,
  },
  tabContentVisible: {
    display: 'flex',
  },
  tabContentHidden: {
    display: 'none',
  },
  tabBar: {
    flexDirection: 'row',
    height: 64,
    paddingHorizontal: 12,
    paddingBottom: 8,
    borderTopWidth: 1,
    alignItems: 'center',
    justifyContent: 'space-around',
  },
  tabBarLight: {
    backgroundColor: '#FFFFFF',
    borderTopColor: 'rgba(21, 33, 29, 0.08)',
  },
  tabBarDark: {
    backgroundColor: '#17231F',
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 6,
    borderRadius: 12,
  },
  tabItemActiveLight: {
    backgroundColor: '#E2F4EE',
  },
  tabItemActiveDark: {
    backgroundColor: 'rgba(32, 90, 70, 0.4)',
  },
  tabGlyph: {
    fontSize: 16,
    marginBottom: 2,
    fontWeight: '800',
  },
  tabGlyphLight: {
    color: '#77736D',
  },
  tabGlyphDark: {
    color: '#929E99',
  },
  tabGlyphActiveLight: {
    color: '#205A46',
  },
  tabGlyphActiveDark: {
    color: '#A8DBC9',
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: '600',
  },
  tabLabelLight: {
    color: '#77736D',
  },
  tabLabelDark: {
    color: '#929E99',
  },
  tabLabelActiveLight: {
    color: '#205A46',
    fontWeight: '800',
  },
  tabLabelActiveDark: {
    color: '#A8DBC9',
    fontWeight: '800',
  },
});
