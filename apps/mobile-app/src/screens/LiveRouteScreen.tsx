import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  ScrollView
} from 'react-native';
import { DataBadge } from '../components/DataBadge';
import { SimulatedActionNotice } from '../components/SimulatedActionNotice';
import { DRIVER_PROFILE, DRIVER_STOPS, DriverStop } from '../data/mockDriverData';

interface LiveRouteScreenProps {
  activeStop?: DriverStop;
  allStops?: DriverStop[];
  onSelectStop?: (stop: DriverStop) => void;
  onStopArrived?: (stop: DriverStop) => Promise<boolean>;
  onReportDelay?: (stopId: string, reason: string, minutes: number) => Promise<boolean>;
  isBackendManifest?: boolean;
}

export const LiveRouteScreen: React.FC<LiveRouteScreenProps> = ({
  activeStop = DRIVER_STOPS[1],
  allStops = DRIVER_STOPS,
  onSelectStop,
  onStopArrived,
  onReportDelay,
  isBackendManifest = false,
}) => {
  const [arrived, setArrived] = useState<boolean>(activeStop.status === 'completed' || activeStop.status === 'in_transit');
  const [showDelayModal, setShowDelayModal] = useState<boolean>(false);
  const [actionNotice, setActionNotice] = useState<{ action: string; detail: string } | null>(null);

  useEffect(() => {
    setArrived(activeStop.status === 'completed' || activeStop.status === 'in_transit');
  }, [activeStop.id, activeStop.status]);

  const handleMarkArrived = async () => {
    const saved = onStopArrived ? await onStopArrived(activeStop) : true;
    if (saved) setArrived(true);
    setActionNotice({
      action: saved ? `Arrival ${isBackendManifest ? 'reported' : 'confirmed'} at ${activeStop.customerName}` : 'Arrival report failed',
      detail: saved
        ? isBackendManifest ? 'Arrival report was accepted by the backend.' : `Stop #${activeStop.sequence} (${activeStop.trackingId}) arrival event recorded in preview memory.`
        : 'The backend did not accept this arrival report. Check the connection and try again.'
    });
  };

  const handleDelaySelect = async (reason: string, minutes: number) => {
    setShowDelayModal(false);
    const saved = onReportDelay ? await onReportDelay(activeStop.id, reason, minutes) : true;
    setActionNotice({
      action: saved ? `Corridor incident reported: +${minutes} min (${reason})` : 'Delay report failed',
      detail: saved
        ? isBackendManifest ? 'Delay report was accepted by the backend for dispatcher review.' : `Preview alert logged for the segment to ${activeStop.customerName}.`
        : 'The backend did not accept this delay report. Check the connection and try again.'
    });
  };

  const handleContactAction = (type: 'call' | 'msg') => {
    if (type === 'call') {
      setActionNotice({
        action: `Calling ${activeStop.contactName}`,
        detail: `Simulated telephony trigger to ${activeStop.contactPhone} (${activeStop.contactRole}). No telephony permission requested.`
      });
    } else {
      setActionNotice({
        action: `Dispatch Alert Sent for Stop #${activeStop.sequence}`,
        detail: `Arrival notice pre-alert transmitted to ${activeStop.consignee}. Local preview channel.`
      });
    }
  };

  const isAtRisk = activeStop.status === 'at_risk';
  const etaDisplay = activeStop.etaInterval.replace(' (simulated)', '');

  return (
    <SafeAreaView style={styles.container}>
      {/* Top HUD Telemetry Bar (Ref 2 Tactical Navigation) */}
      <View style={styles.topHud}>
        <View style={styles.hudLeft}>
            <Text style={styles.hudSub}>{isBackendManifest ? 'ASSIGNED MANIFEST · NAVIGATION UNAVAILABLE' : 'TACTICAL NAVIGATION · BENGALURU'}</Text>
          <Text style={styles.hudLocation} numberOfLines={1}>
            Indiranagar Hub → Stop #{activeStop.sequence} {activeStop.customerName.split(' ')[0]}
          </Text>
        </View>
        <View style={styles.topHudRight}>
          <Text style={styles.hudBattery}>{isBackendManifest ? 'Telemetry unavailable' : `${DRIVER_PROFILE.batteryPct}% SoC`}</Text>
          <DataBadge label={isBackendManifest ? 'Backend · synthetic plan' : 'Preview'} tone={isBackendManifest ? 'warning' : 'preview'} />
        </View>
      </View>

      {/* Stop Sequence Selector Strip (Allows inspecting corridor stops) */}
      <View style={styles.stopSelectorBar}>
        {allStops.map((s) => {
          const isSelected = s.id === activeStop.id;
          const isDone = s.status === 'completed';
          const isRisk = s.status === 'at_risk';
          return (
            <TouchableOpacity
              key={s.id}
              style={[
                styles.stopSelectorTab,
                isSelected && styles.stopSelectorTabActive,
                isDone && styles.stopSelectorTabDone,
                isRisk && !isSelected && styles.stopSelectorTabRisk
              ]}
              onPress={() => onSelectStop && onSelectStop(s)}
              activeOpacity={0.7}
            >
              <Text
                style={[
                  styles.stopSelectorNum,
                  isSelected && styles.stopSelectorNumActive,
                  isDone && styles.stopSelectorNumDone
                ]}
              >
                {isDone ? '✓' : `#${s.sequence}`}
              </Text>
              <Text
                style={[
                  styles.stopSelectorLabel,
                  isSelected && styles.stopSelectorLabelActive
                ]}
                numberOfLines={1}
              >
                {s.customerName.split(' ')[0]}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Action Notice Overlay */}
      {actionNotice && (
        <View style={styles.noticeWrapper}>
          <SimulatedActionNotice
            actionName={actionNotice.action}
            detail={actionNotice.detail}
            onDismiss={() => setActionNotice(null)}
            dark
          />
        </View>
      )}

      {/* Dynamic Turn-by-Turn Maneuver Banner */}
      <View style={styles.maneuverCard}>
        <View style={styles.turnIconCircle}>
          <Text style={styles.turnIconText}>{activeStop.maneuverIcon}</Text>
        </View>
        <View style={styles.maneuverTextWrap}>
          <Text style={styles.maneuverDistance}>{activeStop.maneuverDistance}</Text>
          <Text style={styles.maneuverInstruction}>{activeStop.maneuverInstruction}</Text>
          <Text style={styles.maneuverNext}>{activeStop.maneuverNext}</Text>
        </View>
        <View style={styles.speedWrap}>
          <Text style={styles.speedNum}>{DRIVER_PROFILE.speedKmh}</Text>
          <Text style={styles.speedUnit}>KM/H</Text>
          <Text style={styles.speedLimit}>Limit {DRIVER_PROFILE.speedLimitKmh}</Text>
        </View>
      </View>

      {/* Tactical Route Map Visual Surface (No GPS dot per Rule 8 & map-mobile) */}
      <View style={styles.mapCanvas}>
        {/* Architectural Night Coordinate Grid */}
        <View style={styles.gridOverlay}>
          {/* Grid lines */}
          <View style={[styles.gridLineH, { top: '25%' }]} />
          <View style={[styles.gridLineH, { top: '50%' }]} />
          <View style={[styles.gridLineH, { top: '75%' }]} />
          <View style={[styles.gridLineV, { left: '33%' }]} />
          <View style={[styles.gridLineV, { left: '66%' }]} />

          {/* Active Navigation Polyline Backing (Glow) */}
          <View style={styles.routePolylineBacking} />
          {/* Active Navigation Polyline Core */}
          <View style={styles.routePolylineCore} />

          {/* Directional Chevrons along Polyline (Ref 6 Tipax) */}
          <View style={[styles.chevronMarker, { left: '32%', top: '56%' }]}>
            <Text style={styles.chevronText}>▲</Text>
          </View>
          <View style={[styles.chevronMarker, { left: '46%', top: '44%' }]}>
            <Text style={styles.chevronText}>▲</Text>
          </View>
          <View style={[styles.chevronMarker, { left: '60%', top: '32%' }]}>
            <Text style={styles.chevronText}>▲</Text>
          </View>

          {/* Planned Origin / Hub Waypoint Marker (Hard Rule 8: Planned Stops Only, No GPS Dot) */}
          <View style={styles.originWaypoint}>
            <View style={styles.originIconCircle}>
              <Text style={styles.originIconText}>⌂</Text>
            </View>
            <View style={styles.originCallout}>
              <Text style={styles.originCalloutLabel}>DEPOT HUB</Text>
              <Text style={styles.originCalloutText}>Indiranagar Dock 3</Text>
            </View>
          </View>

          {/* Target Destination Stop Pin (Ref 6 Tipax) */}
          <View style={styles.destPin}>
            <View style={[styles.destPinIconWrap, isAtRisk && styles.destPinIconWrapRisk]}>
              <Text style={styles.destPinIcon}>◎</Text>
            </View>
            <View style={styles.destPinCallout}>
              <Text style={styles.destPinStopNum}>TARGET STOP #{activeStop.sequence}</Text>
              <Text style={styles.destPinName}>{activeStop.customerName.split(' ')[0]}</Text>
            </View>
          </View>

          {/* Subsequent Planned Stop Waypoint (Corridor Preview) */}
          <View style={styles.nextStopWaypoint}>
            <View style={styles.nextStopDot}>
              <Text style={styles.nextStopDotText}>3</Text>
            </View>
            <View style={styles.nextStopCallout}>
              <Text style={styles.nextStopCalloutText}>Next: HSR Bay</Text>
            </View>
          </View>

          {/* Dynamic Corridor Congestion Alert Tag (Ref 2 & Ref 5) */}
          <View style={[styles.trafficAlertTag, isAtRisk && styles.trafficAlertTagRisk]}>
            <Text style={[styles.trafficAlertDot, isAtRisk && styles.trafficAlertDotRisk]}>●</Text>
            <Text style={[styles.trafficAlertText, isAtRisk && styles.trafficAlertTextRisk]}>
              {activeStop.corridorAdvisory || 'Normal corridor traffic · P(late): ' + activeStop.pLate}
            </Text>
          </View>

          {/* Hard Rule 8 Governance Indicator */}
          <View style={styles.ruleNoticeTag}>
            <Text style={styles.ruleNoticeText}>
              Planned Stops & Corridor · No GPS Tracking (Rule 8)
            </Text>
          </View>
        </View>
      </View>

      {/* Delay Options Modal / Drawer */}
      {showDelayModal && (
        <View style={styles.delayPickerSheet}>
          <View style={styles.delayPickerHeader}>
            <Text style={styles.delayPickerTitle}>Report Corridor Incident</Text>
            <TouchableOpacity
              onPress={() => setShowDelayModal(false)}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
            >
              <Text style={styles.delayPickerClose}>✕</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.delayPickerSub}>
            {isBackendManifest ? 'Select the delay category to report to dispatch:' : 'Select incident to queue simulated telemetry to dispatcher:'}
          </Text>
          <View style={styles.delayOptionsRow}>
            <TouchableOpacity
              style={styles.delayOptionBtn}
              onPress={() => handleDelaySelect('Heavy Corridor Traffic', 15)}
            >
              <Text style={styles.delayOptionText}>Traffic Jam (+15m)</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.delayOptionBtn}
              onPress={() => handleDelaySelect('Dock Bay Queue', 20)}
            >
              <Text style={styles.delayOptionText}>Dock Queue (+20m)</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.delayOptionBtn}
              onPress={() => handleDelaySelect('Road Construction Detour', 10)}
            >
              <Text style={styles.delayOptionText}>Detour (+10m)</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Bottom Destination HUD & Active Delivery Actions */}
      <View style={styles.bottomSheet}>
        <View style={styles.sheetHandle} />

        {/* Destination Header */}
        <View style={styles.destinationHeader}>
          <View style={styles.destHeaderLeft}>
            <Text style={styles.destinationStopIdx}>
              STOP #{activeStop.sequence} OF {allStops.length} · {activeStop.trackingId}
            </Text>
            <Text style={styles.destinationName}>{activeStop.customerName}</Text>
            <Text style={styles.destinationAddress}>{activeStop.address}</Text>
          </View>

          {/* Simulated ETA Interval (Honest Data States §2 & Hard Rule 6) */}
          <View style={styles.etaBadge}>
            <Text style={styles.etaTitle}>{isBackendManifest ? 'SCENARIO ETA' : 'SIMULATED ETA'}</Text>
            <Text style={styles.etaText}>{etaDisplay}</Text>
            <Text style={styles.etaSub}>{isBackendManifest ? 'not navigation timing' : '(simulated)'}</Text>
          </View>
        </View>

        {/* Cargo & Cold-Chain Telemetry Row */}
        <View style={styles.metaRow}>
          <View style={styles.metaPill}>
            <Text style={styles.metaPillText}>{isBackendManifest ? 'Load details unavailable' : `Demand: ${activeStop.demandKg} kg (${activeStop.packageCount} pkgs)`}</Text>
          </View>
          <View style={styles.metaPill}>
            <Text style={styles.metaPillText}>Window: {activeStop.windowStart} – {activeStop.windowEnd}</Text>
          </View>
          <View style={styles.metaPill}>
            <Text style={styles.metaPillHighlight}>{activeStop.specialHandling}</Text>
          </View>
        </View>

        {/* Ref 6 Tipax Driver Contact Row */}
        {!isBackendManifest && <View style={styles.contactBar}>
          <View style={styles.contactInfoWrap}>
            <Text style={styles.contactNameText}>{activeStop.contactName}</Text>
            <Text style={styles.contactRoleText}>{activeStop.contactRole} · {activeStop.contactPhone}</Text>
          </View>
          <View style={styles.contactBtnGroup}>
            <TouchableOpacity
              style={styles.contactActionBtn}
              onPress={() => handleContactAction('call')}
              activeOpacity={0.7}
              hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}
            >
              <Text style={styles.contactActionBtnText}>Call</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.contactActionBtn}
              onPress={() => handleContactAction('msg')}
              activeOpacity={0.7}
              hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}
            >
              <Text style={styles.contactActionBtnText}>Message</Text>
            </TouchableOpacity>
          </View>
        </View>}

        {/* Tactical Primary & Secondary CTAs */}
        <View style={styles.ctaRow}>
          <TouchableOpacity
            style={[styles.primaryCta, arrived && styles.primaryCtaDone]}
            onPress={handleMarkArrived}
            disabled={arrived}
            activeOpacity={0.8}
          >
            <Text style={styles.primaryCtaText}>
              {arrived ? '✓ Arrival Confirmed at Dock' : 'Arrived at Stop'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryCta}
            onPress={() => setShowDelayModal(true)}
            activeOpacity={0.8}
          >
            <Text style={styles.secondaryCtaText}>Report Delay</Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F1713',
  },
  topHud: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.08)',
  },
  hudLeft: {
    flex: 1,
    marginRight: 10,
  },
  hudSub: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#A8DBC9',
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  hudLocation: {
    fontSize: 13,
    color: '#FFFFFF',
    fontWeight: '800',
    marginTop: 2,
  },
  topHudRight: {
    alignItems: 'flex-end',
    gap: 4,
  },
  hudBattery: {
    fontSize: 11,
    color: '#A8DBC9',
    fontWeight: '700',
    fontFamily: 'monospace',
  },
  stopSelectorBar: {
    flexDirection: 'row',
    backgroundColor: '#17231F',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.06)',
    gap: 8,
  },
  stopSelectorTab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 5,
    paddingHorizontal: 6,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    gap: 4,
  },
  stopSelectorTabActive: {
    backgroundColor: '#205A46',
    borderWidth: 1,
    borderColor: '#A8DBC9',
  },
  stopSelectorTabDone: {
    backgroundColor: 'rgba(32, 90, 70, 0.35)',
  },
  stopSelectorTabRisk: {
    backgroundColor: 'rgba(198, 40, 40, 0.25)',
  },
  stopSelectorNum: {
    fontSize: 10,
    fontWeight: '800',
    fontFamily: 'monospace',
    color: '#929E99',
  },
  stopSelectorNumActive: {
    color: '#FFFFFF',
  },
  stopSelectorNumDone: {
    color: '#A8DBC9',
  },
  stopSelectorLabel: {
    fontSize: 10,
    fontWeight: '600',
    color: '#929E99',
  },
  stopSelectorLabelActive: {
    color: '#FFFFFF',
    fontWeight: '700',
  },
  noticeWrapper: {
    paddingHorizontal: 14,
  },
  maneuverCard: {
    marginHorizontal: 14,
    marginTop: 8,
    marginBottom: 6,
    backgroundColor: '#17231F',
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(168, 219, 201, 0.25)',
    flexDirection: 'row',
    alignItems: 'center',
  },
  turnIconCircle: {
    width: 36,
    height: 36,
    borderRadius: 12,
    backgroundColor: '#205A46',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  turnIconText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '900',
  },
  maneuverTextWrap: {
    flex: 1,
  },
  maneuverDistance: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#A8DBC9',
    fontWeight: '700',
  },
  maneuverInstruction: {
    fontSize: 13,
    fontWeight: '800',
    color: '#FFFFFF',
    marginTop: 1,
  },
  maneuverNext: {
    fontSize: 10,
    color: '#929E99',
    marginTop: 2,
  },
  speedWrap: {
    alignItems: 'center',
    borderLeftWidth: 1,
    borderLeftColor: 'rgba(255, 255, 255, 0.1)',
    paddingLeft: 12,
  },
  speedNum: {
    fontSize: 20,
    fontWeight: '900',
    color: '#FFFFFF',
  },
  speedUnit: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#A8DBC9',
  },
  speedLimit: {
    fontSize: 10,
    color: '#A8DBC9',
    marginTop: 2,
  },
  mapCanvas: {
    flex: 1,
    backgroundColor: '#121C17',
    position: 'relative',
    overflow: 'hidden',
    minHeight: 180,
  },
  gridOverlay: {
    ...StyleSheet.absoluteFill,
  },
  gridLineH: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
  },
  gridLineV: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
  },
  routePolylineBacking: {
    position: 'absolute',
    left: '20%',
    top: '28%',
    width: 250,
    height: 8,
    backgroundColor: 'rgba(32, 90, 70, 0.45)',
    transform: [{ rotate: '-38deg' }],
    borderRadius: 4,
  },
  routePolylineCore: {
    position: 'absolute',
    left: '20%',
    top: '29%',
    width: 250,
    height: 4,
    backgroundColor: '#205A46',
    transform: [{ rotate: '-38deg' }],
    borderRadius: 2,
  },
  chevronMarker: {
    position: 'absolute',
    transform: [{ rotate: '52deg' }],
  },
  chevronText: {
    color: '#A8DBC9',
    fontSize: 11,
    fontWeight: '900',
  },
  originWaypoint: {
    position: 'absolute',
    left: '14%',
    top: '64%',
    alignItems: 'center',
  },
  originIconCircle: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: '#1E3328',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#A8DBC9',
  },
  originIconText: {
    color: '#A8DBC9',
    fontSize: 12,
    fontWeight: '900',
  },
  originCallout: {
    backgroundColor: '#17231F',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginTop: 4,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
    alignItems: 'center',
  },
  originCalloutLabel: {
    color: '#A8DBC9',
    fontSize: 9,
    fontFamily: 'monospace',
    fontWeight: '800',
  },
  originCalloutText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '600',
  },
  destPin: {
    position: 'absolute',
    left: '68%',
    top: '20%',
    alignItems: 'center',
  },
  destPinIconWrap: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: '#205A46',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  destPinIconWrapRisk: {
    backgroundColor: '#C62828',
  },
  destPinIcon: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  destPinCallout: {
    backgroundColor: '#17231F',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    marginTop: 3,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
  },
  destPinStopNum: {
    color: '#A8DBC9',
    fontSize: 9,
    fontFamily: 'monospace',
    fontWeight: '800',
  },
  destPinName: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
  },
  nextStopWaypoint: {
    position: 'absolute',
    left: '82%',
    top: '46%',
    alignItems: 'center',
    opacity: 0.7,
  },
  nextStopDot: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: '#3E4D45',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#A8DBC9',
  },
  nextStopDotText: {
    color: '#FFFFFF',
    fontSize: 9,
    fontWeight: '800',
  },
  nextStopCallout: {
    backgroundColor: '#17231F',
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 4,
    marginTop: 2,
  },
  nextStopCalloutText: {
    color: '#929E99',
    fontSize: 9,
    fontWeight: '600',
  },
  trafficAlertTag: {
    position: 'absolute',
    top: 10,
    left: 14,
    right: 14,
    backgroundColor: 'rgba(32, 90, 70, 0.25)',
    borderWidth: 1,
    borderColor: '#205A46',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  trafficAlertTagRisk: {
    backgroundColor: 'rgba(215, 138, 69, 0.22)',
    borderColor: '#D78A45',
  },
  trafficAlertDot: {
    color: '#A8DBC9',
    fontSize: 10,
  },
  trafficAlertDotRisk: {
    color: '#D78A45',
  },
  trafficAlertText: {
    color: '#D9E6DC',
    fontSize: 10,
    fontWeight: '700',
    flex: 1,
  },
  trafficAlertTextRisk: {
    color: '#FDF4EB',
  },
  ruleNoticeTag: {
    position: 'absolute',
    bottom: 8,
    left: 14,
    right: 14,
    backgroundColor: 'rgba(15, 23, 19, 0.85)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    alignItems: 'center',
  },
  ruleNoticeText: {
    fontSize: 9,
    fontFamily: 'monospace',
    color: '#929E99',
  },
  delayPickerSheet: {
    backgroundColor: '#1B2924',
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    padding: 14,
    borderTopWidth: 1,
    borderTopColor: '#D78A45',
  },
  delayPickerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  delayPickerTitle: {
    fontSize: 13,
    fontWeight: '800',
    color: '#FFFFFF',
  },
  delayPickerClose: {
    color: '#929E99',
    fontSize: 16,
    fontWeight: '700',
    padding: 4,
  },
  delayPickerSub: {
    fontSize: 11,
    color: '#A0B2AA',
    marginBottom: 10,
  },
  delayOptionsRow: {
    flexDirection: 'row',
    gap: 8,
  },
  delayOptionBtn: {
    flex: 1,
    backgroundColor: 'rgba(215, 138, 69, 0.2)',
    paddingVertical: 9,
    paddingHorizontal: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(215, 138, 69, 0.5)',
    alignItems: 'center',
  },
  delayOptionText: {
    color: '#FDF4EB',
    fontSize: 10,
    fontWeight: '700',
    textAlign: 'center',
  },
  bottomSheet: {
    backgroundColor: '#17231F',
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    padding: 14,
    paddingBottom: 20,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  sheetHandle: {
    width: 36,
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: 10,
  },
  destinationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  destHeaderLeft: {
    flex: 1,
    marginRight: 10,
  },
  destinationStopIdx: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#A8DBC9',
    fontWeight: '800',
  },
  destinationName: {
    fontSize: 15,
    fontWeight: '800',
    color: '#FFFFFF',
    marginTop: 2,
  },
  destinationAddress: {
    fontSize: 11,
    color: '#929E99',
    marginTop: 1,
  },
  etaBadge: {
    backgroundColor: 'rgba(32, 90, 70, 0.35)',
    borderWidth: 1,
    borderColor: '#205A46',
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 5,
    alignItems: 'center',
  },
  etaTitle: {
    fontSize: 9,
    fontFamily: 'monospace',
    color: '#A8DBC9',
    fontWeight: '800',
  },
  etaText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
    marginTop: 2,
  },
  etaSub: {
    color: '#929E99',
    fontSize: 9,
  },
  metaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 10,
  },
  metaPill: {
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderRadius: 4,
  },
  metaPillText: {
    color: '#D9E6DC',
    fontSize: 10,
    fontWeight: '600',
  },
  metaPillHighlight: {
    color: '#A8DBC9',
    fontSize: 10,
    fontWeight: '700',
  },
  contactBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 8,
    marginTop: 10,
  },
  contactInfoWrap: {
    flex: 1,
    marginRight: 8,
  },
  contactNameText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  contactRoleText: {
    fontSize: 10,
    color: '#929E99',
    marginTop: 1,
  },
  contactBtnGroup: {
    flexDirection: 'row',
    gap: 6,
  },
  contactActionBtn: {
    backgroundColor: 'rgba(255, 255, 255, 0.12)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 6,
  },
  contactActionBtnText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '700',
  },
  ctaRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 12,
  },
  primaryCta: {
    flex: 2,
    backgroundColor: '#205A46',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryCtaDone: {
    backgroundColor: '#1E4B3B',
  },
  primaryCtaText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
  secondaryCta: {
    flex: 1,
    backgroundColor: 'rgba(215, 138, 69, 0.2)',
    borderWidth: 1,
    borderColor: '#D78A45',
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryCtaText: {
    color: '#FDF4EB',
    fontSize: 12,
    fontWeight: '700',
  },
});
