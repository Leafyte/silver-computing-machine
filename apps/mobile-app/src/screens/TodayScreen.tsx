import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  SafeAreaView
} from 'react-native';
import { DataBadge } from '../components/DataBadge';
import { SimulatedActionNotice } from '../components/SimulatedActionNotice';
import { DriverIdentity } from '../api/driverClient';
import {
  DRIVER_PROFILE,
  SHIFT_DATES,
  SHIFT_SCHEDULES,
  DriverStop
} from '../data/mockDriverData';

interface TodayScreenProps {
  stops: DriverStop[];
  onToggleComplete: (stop: DriverStop) => Promise<boolean>;
  onNavigateToLiveRoute: (stop: DriverStop) => void;
  selectedDate: string;
  onSelectDate: (date: string) => void;
  shiftActive: boolean;
  onToggleShift: () => void;
  driverIdentity: DriverIdentity | null;
}

export const TodayScreen: React.FC<TodayScreenProps> = ({
  stops,
  onToggleComplete,
  onNavigateToLiveRoute,
  selectedDate,
  onSelectDate,
  shiftActive,
  onToggleShift,
  driverIdentity,
}) => {
  const [actionNotice, setActionNotice] = useState<{ action: string; detail: string } | null>(null);

  const currentSchedule = SHIFT_SCHEDULES[selectedDate] || SHIFT_SCHEDULES['29 Oct'];
  const displayStops = driverIdentity || selectedDate === '29 Oct' ? stops : currentSchedule.stops;
  const isToday = selectedDate === '29 Oct';

  const completedCount = displayStops.filter(s => s.status === 'completed').length;
  const progressPct = displayStops.length > 0 ? Math.round((completedCount / displayStops.length) * 100) : 0;

  const handleContactAction = (type: 'call' | 'msg', stop: DriverStop) => {
    if (type === 'call') {
      setActionNotice({
        action: `Contacting ${stop.contactName}`,
        detail: `Simulated phone trigger to ${stop.contactPhone} (${stop.contactRole} at ${stop.customerName}). No cellular permission requested.`
      });
    } else {
      setActionNotice({
        action: `Dispatch Message Drafted for ${stop.customerName}`,
        detail: `Delivery ETA notification queued for ${stop.contactName}. Local preview message channel only.`
      });
    }
  };

  const handleCompletePress = async (stop: DriverStop) => {
    if (driverIdentity && stop.status === 'completed') return;
    const saved = await onToggleComplete(stop);
    const isNowDone = stop.status !== 'completed';
    setActionNotice({
      action: saved ? `Stop #${stop.sequence} ${isNowDone ? 'Marked Complete' : 'Re-opened'}` : 'Stop update failed',
      detail: saved
        ? `${stop.customerName} status ${driverIdentity ? 'saved to the backend' : 'updated in local preview state'}.`
        : 'The backend did not accept this stop report. Review the connection and try again.'
    });
  };

  const handleShiftPress = () => {
    onToggleShift();
    setActionNotice({
      action: !shiftActive ? 'Shift Resumed (Online)' : 'Shift Paused (Standby)',
      detail: `Operator ${DRIVER_PROFILE.name} toggled shift state to ${
        !shiftActive ? 'Active' : 'Standby'
      }. Local preview telemetry updated.`
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Top Header & Sentence-Case Source Badge */}
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>01 · DISPATCH MANIFEST</Text>
            <Text style={styles.greeting}>
              {driverIdentity ? `Driver ${driverIdentity.driver_id} · ${driverIdentity.vehicle_id}` : `${currentSchedule.shiftId} · ${DRIVER_PROFILE.name}`}
            </Text>
            {!driverIdentity && <Text style={styles.shiftHoursText}>{currentSchedule.shiftHours}</Text>}
          </View>
          <DataBadge
            label={driverIdentity ? 'Backend manifest' : isToday ? 'Preview' : 'Preview · Upcoming'}
            tone={driverIdentity ? 'success' : 'preview'}
          />
        </View>

        {/* Global Action Feedback Notice (Surfaced on Unsent Actions) */}
        {actionNotice && (
          <SimulatedActionNotice
            actionName={actionNotice.action}
            detail={actionNotice.detail}
            onDismiss={() => setActionNotice(null)}
          />
        )}

        {/* Date Selector Carousel (Vanta Ride Style Horizontal Strip) */}
        {!driverIdentity && <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.dateCarousel}
        >
          {SHIFT_DATES.map(d => {
            const isSelected = selectedDate === d.date;
            return (
              <TouchableOpacity
                key={d.date}
                style={[styles.dateCard, isSelected && styles.dateCardActive]}
                onPress={() => onSelectDate(d.date)}
                activeOpacity={0.7}
              >
                <Text style={[styles.dateDay, isSelected && styles.dateDayActive]}>{d.day}</Text>
                <Text style={[styles.dateNum, isSelected && styles.dateNumActive]}>{d.date.split(' ')[0]}</Text>
                <Text style={[styles.dateSub, isSelected && styles.dateSubActive]}>{d.stops} stops</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>}

        {/* Daily Summary & Earnings Card (Vanta Ride Clean Spacing & HUD Telemetry) */}
        <View style={styles.summaryCard}>
          <View style={styles.summaryTop}>
            <View>
              <Text style={styles.summaryLabel}>{driverIdentity ? 'ASSIGNED MANIFEST' : isToday ? "TODAY'S ESTIMATED EARNINGS" : 'SCHEDULED SHIFT PAY'}</Text>
              <Text style={styles.summaryAmount}>{driverIdentity ? `${displayStops.length} stops` : `₹${currentSchedule.earningsInr.toLocaleString('en-IN')}`}</Text>
              {!driverIdentity && <Text style={styles.assumedRate}>{DRIVER_PROFILE.assumedRateLabel}</Text>}
            </View>

            {/* Interactive Shift Toggle */}
            {!driverIdentity && <TouchableOpacity
              style={[styles.shiftBadge, shiftActive ? styles.shiftBadgeActive : styles.shiftBadgeStandby]}
              onPress={handleShiftPress}
              activeOpacity={0.8}
            >
              <View style={[styles.statusDot, { backgroundColor: shiftActive ? '#205A46' : '#77736D' }]} />
              <Text style={[styles.shiftBadgeText, shiftActive ? styles.shiftBadgeTextActive : styles.shiftBadgeTextStandby]}>
                {shiftActive ? 'SHIFT ACTIVE' : 'STANDBY'}
              </Text>
            </TouchableOpacity>}
            {driverIdentity && <Text style={styles.assumedRate}>Shift state is not exposed by the backend.</Text>}
          </View>

          {/* Stop Delivery Progress Bar */}
          <View style={styles.progressContainer}>
            <View style={styles.progressLabels}>
              <Text style={styles.progressText}>Stop Delivery Progress</Text>
              <Text style={styles.progressPctText}>{completedCount}/{displayStops.length} ({progressPct}%)</Text>
            </View>
            <View style={styles.progressBarTrack}>
              <View style={[styles.progressBarFill, { width: `${progressPct}%` }]} />
            </View>
          </View>

          {/* Vehicle Telemetry Strip */}
          <View style={styles.vehicleStrip}>
            <View>
              <Text style={styles.vehicleInfoText}>Unit {DRIVER_PROFILE.vehicleCallsign} · {DRIVER_PROFILE.vehicleModel}</Text>
              <Text style={styles.vehicleSubText}>Payload: {DRIVER_PROFILE.currentPayloadKg}/{DRIVER_PROFILE.maxPayloadKg} kg</Text>
            </View>
            <View style={styles.telemetryRight}>
              <Text style={styles.batteryText}>{DRIVER_PROFILE.batteryPct}% SoC</Text>
              <Text style={styles.speedText}>{DRIVER_PROFILE.speedKmh} km/h</Text>
            </View>
          </View>
        </View>

        {/* Section Header */}
        <View style={styles.sectionHeader}>
          <View>
            <Text style={styles.sectionTitle}>
              Stop Sequence & Timeline ({displayStops.length})
            </Text>
            <Text style={styles.sectionSub}>Select stop to enter tactical corridor navigation</Text>
          </View>
          <DataBadge label="Simulated · held-out run" tone="preview" />
        </View>

        {/* Vertical Stop Itinerary Cards with Dotted Route Timeline (Vanta Ride & Tipax Style) */}
        <View style={styles.timelineContainer}>
          {displayStops.map((stop, index) => {
            const isCompleted = stop.status === 'completed';
            const isInTransit = stop.status === 'in_transit';
            const isAtRisk = stop.status === 'at_risk';
            const isLast = index === displayStops.length - 1;

            return (
              <View key={stop.id} style={styles.timelineRow}>
                {/* Left Vertical Dotted Timeline Rail (Ref 3 Vanta Ride) */}
                <View style={styles.timelineRail}>
                  <View
                    style={[
                      styles.timelineNode,
                      isCompleted && styles.timelineNodeDone,
                      isInTransit && styles.timelineNodeTransit,
                      isAtRisk && styles.timelineNodeRisk
                    ]}
                  >
                    {isCompleted ? (
                      <Text style={styles.timelineNodeDoneText}>✓</Text>
                    ) : (
                      <Text style={[styles.timelineNodeText, isInTransit && styles.timelineNodeTransitText]}>
                        {stop.sequence}
                      </Text>
                    )}
                  </View>

                  {!isLast && (
                    <View style={styles.verticalDottedLineContainer}>
                      <View style={[styles.verticalDottedDash, isCompleted && styles.verticalDottedDashDone]} />
                      <View style={[styles.verticalDottedDash, isCompleted && styles.verticalDottedDashDone]} />
                      <View style={[styles.verticalDottedDash, isCompleted && styles.verticalDottedDashDone]} />
                      <View style={[styles.verticalDottedDash, isCompleted && styles.verticalDottedDashDone]} />
                      <View style={[styles.verticalDottedDash, isCompleted && styles.verticalDottedDashDone]} />
                    </View>
                  )}
                </View>

                {/* Right Stop Itinerary Card */}
                <View style={[styles.stopCard, isCompleted && styles.stopCardCompleted]}>
                  {/* Card Header: Stop Sequence, Locked Prefix Badge & Status Pill */}
                  <View style={styles.stopCardHeader}>
                    <View style={styles.cardHeaderLeft}>
                      <View style={styles.stopSeqBadge}>
                        <Text style={styles.stopSeqBadgeText}>STOP #{stop.sequence}</Text>
                      </View>
                      {stop.isLocked && (
                        <View style={styles.lockedBadge}>
                          <Text style={styles.lockedBadgeText}>🔒 Locked Prefix</Text>
                        </View>
                      )}
                    </View>

                    <View
                      style={[
                        styles.statusPill,
                        isCompleted && styles.pillSuccess,
                        isInTransit && styles.pillTransit,
                        isAtRisk && styles.pillRisk
                      ]}
                    >
                      <Text
                        style={[
                          styles.statusPillText,
                          isCompleted && styles.pillTextSuccess,
                          isInTransit && styles.pillTextTransit,
                          isAtRisk && styles.pillTextRisk
                        ]}
                      >
                        {stop.status.replace('_', ' ').toUpperCase()}
                      </Text>
                    </View>
                  </View>

                  {/* Customer & Consignment Info (Ref 6 Tipax) */}
                  <View style={styles.customerBlock}>
                    <Text style={[styles.stopCustomerName, isCompleted && styles.textMuted]}>
                      {stop.customerName}
                    </Text>
                    <Text style={styles.stopConsignee}>{stop.consignee}</Text>
                    <Text style={styles.stopAddress}>{stop.address}</Text>
                  </View>

                  {/* Shipment Identification & Tracking Tag (Ref 6 Tipax) */}
                  <View style={styles.shipmentMetaRow}>
                    <View style={styles.metaTag}>
                      <Text style={styles.metaTagMono}>{stop.trackingId}</Text>
                    </View>
                    <View style={styles.metaTag}>
                      <Text style={styles.metaTagText}>{stop.packageCount} pkgs · {stop.demandKg} kg</Text>
                    </View>
                    <View style={styles.metaTag}>
                      <Text style={styles.metaTagText}>{stop.specialHandling}</Text>
                    </View>
                  </View>

                  {/* Timing & Delay Risk Metrics (Honest Data States §2) */}
                  <View style={styles.metricsBox}>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricLabel}>DELIVERY WINDOW</Text>
                      <Text style={styles.metricValue}>{stop.windowStart} – {stop.windowEnd}</Text>
                    </View>

                    <View style={styles.metricItem}>
                      <Text style={styles.metricLabel}>SIMULATED ETA</Text>
                      <Text style={styles.metricValue}>{stop.etaInterval}</Text>
                    </View>

                    <View style={styles.metricItem}>
                      <Text style={styles.metricLabel}>P(LATE)</Text>
                      <Text style={[styles.metricValue, isAtRisk && styles.textRisk]}>
                        {stop.pLate}
                      </Text>
                    </View>
                  </View>

                  {/* Corridor Delay Alert if present */}
                  {stop.corridorAdvisory && (
                    <View style={[styles.advisoryBanner, isAtRisk && styles.advisoryBannerRisk]}>
                      <Text style={[styles.advisoryText, isAtRisk && styles.advisoryTextRisk]}>
                        {stop.corridorAdvisory}
                      </Text>
                    </View>
                  )}

                  {/* Quick Contact Actions (Ref 6 Tipax) */}
                  {!driverIdentity ? <View style={styles.contactRow}>
                    <View style={styles.contactInfo}>
                      <Text style={styles.contactName}>{stop.contactName}</Text>
                      <Text style={styles.contactRole}>{stop.contactRole}</Text>
                    </View>
                    <View style={styles.contactActions}>
                      <TouchableOpacity
                        style={styles.contactBtn}
                        onPress={() => handleContactAction('call', stop)}
                        activeOpacity={0.7}
                        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                      >
                        <Text style={styles.contactBtnText}>Call</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={styles.contactBtn}
                        onPress={() => handleContactAction('msg', stop)}
                        activeOpacity={0.7}
                        hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                      >
                        <Text style={styles.contactBtnText}>Message</Text>
                      </TouchableOpacity>
                    </View>
                  </View> : <Text style={styles.contactRole}>Consignee contact data is not included in the manifest.</Text>}

                  {/* Primary & Secondary Action Buttons */}
                  <View style={styles.cardActionsRow}>
                    <TouchableOpacity
                      style={styles.navButton}
                      onPress={() => onNavigateToLiveRoute(stop)}
                      activeOpacity={0.8}
                    >
                      <Text style={styles.navButtonText}>{driverIdentity ? 'View route details →' : 'Corridor Navigation →'}</Text>
                    </TouchableOpacity>

                    <TouchableOpacity
                      style={[styles.checkButton, isCompleted && styles.checkButtonDone]}
                      onPress={() => handleCompletePress(stop)}
                      activeOpacity={0.8}
                      hitSlop={{ top: 4, bottom: 4, left: 4, right: 4 }}
                    >
                      <Text style={[styles.checkButtonText, isCompleted && styles.checkButtonTextDone]}>
                      {isCompleted ? '✓ Done' : driverIdentity ? 'Report complete' : 'Complete'}
                      </Text>
                    </TouchableOpacity>
                  </View>
                </View>
              </View>
            );
          })}
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
  greeting: {
    fontSize: 20,
    fontWeight: '800',
    color: '#15211D',
    marginTop: 2,
  },
  shiftHoursText: {
    fontSize: 11,
    color: '#55534F',
    fontWeight: '600',
    marginTop: 2,
  },
  dateCarousel: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
    paddingVertical: 2,
  },
  dateCard: {
    width: 80,
    paddingVertical: 10,
    paddingHorizontal: 8,
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
    alignItems: 'center',
  },
  dateCardActive: {
    backgroundColor: '#17231F',
    borderColor: '#17231F',
  },
  dateDay: {
    fontSize: 11,
    fontWeight: '600',
    color: '#77736D',
  },
  dateDayActive: {
    color: '#A8DBC9',
  },
  dateNum: {
    fontSize: 16,
    fontWeight: '800',
    color: '#15211D',
    marginTop: 2,
  },
  dateNumActive: {
    color: '#FFFFFF',
  },
  dateSub: {
    fontSize: 10,
    color: '#929E99',
    marginTop: 2,
  },
  dateSubActive: {
    color: '#A8DBC9',
  },
  summaryCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
    marginBottom: 20,
    shadowColor: '#101828',
    shadowOpacity: 0.03,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
  },
  summaryTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  summaryLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: '#77736D',
    fontFamily: 'monospace',
  },
  summaryAmount: {
    fontSize: 26,
    fontWeight: '800',
    color: '#205A46',
    marginTop: 2,
  },
  assumedRate: {
    fontSize: 10,
    color: '#77736D',
    marginTop: 2,
  },
  shiftBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
    gap: 6,
  },
  shiftBadgeActive: {
    backgroundColor: '#E2F4EE',
    borderColor: '#A8DBC9',
  },
  shiftBadgeStandby: {
    backgroundColor: '#F3F4F6',
    borderColor: '#E5E7EB',
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 999,
  },
  shiftBadgeText: {
    fontSize: 10,
    fontWeight: '800',
  },
  shiftBadgeTextActive: {
    color: '#205A46',
  },
  shiftBadgeTextStandby: {
    color: '#77736D',
  },
  progressContainer: {
    marginTop: 16,
  },
  progressLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  progressText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#15211D',
  },
  progressPctText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#205A46',
  },
  progressBarTrack: {
    height: 6,
    backgroundColor: '#EEF2ED',
    borderRadius: 999,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#205A46',
    borderRadius: 999,
  },
  vehicleStrip: {
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(21, 33, 29, 0.06)',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  vehicleInfoText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#15211D',
  },
  vehicleSubText: {
    fontSize: 10,
    color: '#77736D',
    marginTop: 1,
  },
  telemetryRight: {
    alignItems: 'flex-end',
  },
  batteryText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#205A46',
  },
  speedText: {
    fontSize: 10,
    color: '#77736D',
    marginTop: 1,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '800',
    color: '#15211D',
  },
  sectionSub: {
    fontSize: 10,
    color: '#77736D',
    marginTop: 1,
  },
  timelineContainer: {
    paddingLeft: 2,
  },
  timelineRow: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  timelineRail: {
    width: 32,
    alignItems: 'center',
    marginRight: 10,
  },
  timelineNode: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#17231F',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  timelineNodeDone: {
    backgroundColor: '#205A46',
  },
  timelineNodeTransit: {
    backgroundColor: '#205A46',
    borderWidth: 2,
    borderColor: '#A8DBC9',
  },
  timelineNodeRisk: {
    backgroundColor: '#C62828',
  },
  timelineNodeText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '800',
  },
  timelineNodeTransitText: {
    color: '#FFFFFF',
  },
  timelineNodeDoneText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  verticalDottedLineContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'space-evenly',
    paddingVertical: 6,
  },
  verticalDottedDash: {
    width: 2,
    height: 10,
    backgroundColor: 'rgba(21, 33, 29, 0.2)',
    borderRadius: 1,
    marginVertical: 2,
  },
  verticalDottedDashDone: {
    backgroundColor: '#A8DBC9',
  },
  stopCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
  },
  stopCardCompleted: {
    backgroundColor: '#FAFAF9',
    borderColor: 'rgba(21, 33, 29, 0.04)',
    opacity: 0.88,
  },
  stopCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  stopSeqBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: '#F3F4F6',
    borderRadius: 4,
  },
  stopSeqBadgeText: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: '800',
    color: '#4B5563',
  },
  lockedBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: '#EFF6FF',
    borderRadius: 4,
    borderWidth: 1,
    borderColor: '#BFDBFE',
  },
  lockedBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#1E40AF',
  },
  statusPill: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
    backgroundColor: '#EEF2ED',
  },
  pillSuccess: {
    backgroundColor: '#E2F4EE',
  },
  pillTransit: {
    backgroundColor: '#E0F2FE',
  },
  pillRisk: {
    backgroundColor: '#FBECEB',
  },
  statusPillText: {
    fontSize: 10,
    fontWeight: '800',
    color: '#4B5563',
  },
  pillTextSuccess: {
    color: '#205A46',
  },
  pillTextTransit: {
    color: '#0369A1',
  },
  pillTextRisk: {
    color: '#C62828',
  },
  customerBlock: {
    marginBottom: 8,
  },
  stopCustomerName: {
    fontSize: 14,
    fontWeight: '800',
    color: '#15211D',
  },
  stopConsignee: {
    fontSize: 11,
    fontWeight: '600',
    color: '#55534F',
    marginTop: 1,
  },
  stopAddress: {
    fontSize: 11,
    color: '#77736D',
    marginTop: 2,
  },
  textMuted: {
    color: '#77736D',
  },
  shipmentMetaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 10,
  },
  metaTag: {
    backgroundColor: '#F4F6F4',
    paddingHorizontal: 6,
    paddingVertical: 3,
    borderRadius: 4,
  },
  metaTagMono: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#205A46',
    fontWeight: '700',
  },
  metaTagText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#55534F',
  },
  metricsBox: {
    backgroundColor: '#F9FAF9',
    borderRadius: 10,
    padding: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.04)',
  },
  metricItem: {
    flex: 1,
  },
  metricLabel: {
    fontSize: 10,
    fontFamily: 'monospace',
    color: '#77736D',
    fontWeight: '700',
  },
  metricValue: {
    fontSize: 10,
    fontWeight: '700',
    color: '#15211D',
    marginTop: 2,
  },
  textRisk: {
    color: '#C62828',
  },
  advisoryBanner: {
    backgroundColor: '#FDF4EB',
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#FBD6B0',
    marginBottom: 10,
  },
  advisoryBannerRisk: {
    backgroundColor: '#FBECEB',
    borderColor: '#F5C2C0',
  },
  advisoryText: {
    fontSize: 10,
    color: '#B26B00',
    fontWeight: '700',
  },
  advisoryTextRisk: {
    color: '#C62828',
  },
  contactRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.05)',
    marginBottom: 10,
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    fontSize: 11,
    fontWeight: '700',
    color: '#15211D',
  },
  contactRole: {
    fontSize: 10,
    color: '#77736D',
  },
  contactActions: {
    flexDirection: 'row',
    gap: 6,
  },
  contactBtn: {
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.15)',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  contactBtnText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#15211D',
  },
  cardActionsRow: {
    flexDirection: 'row',
    gap: 8,
  },
  navButton: {
    flex: 1,
    backgroundColor: '#17231F',
    paddingVertical: 9,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  navButtonText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
  },
  checkButton: {
    paddingHorizontal: 14,
    paddingVertical: 9,
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.15)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkButtonDone: {
    backgroundColor: '#E2F4EE',
    borderColor: '#A8DBC9',
  },
  checkButtonText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#15211D',
  },
  checkButtonTextDone: {
    color: '#205A46',
  },
});
