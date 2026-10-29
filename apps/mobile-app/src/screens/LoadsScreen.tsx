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
import {
  DOCK_CHECKLIST,
  DockChecklistItem,
  DRIVER_PROFILE,
  MANIFEST_CONSIGNMENTS
} from '../data/mockDriverData';

interface LoadsScreenProps {
  checklist?: DockChecklistItem[];
  onToggleCheck?: (id: string) => void;
  backendManifestActive?: boolean;
}

export const LoadsScreen: React.FC<LoadsScreenProps> = ({
  checklist: externalChecklist,
  onToggleCheck: externalOnToggleCheck,
  backendManifestActive = false,
}) => {
  const [internalChecklist, setInternalChecklist] = useState<DockChecklistItem[]>(DOCK_CHECKLIST);
  const [actionNotice, setActionNotice] = useState<{ action: string; detail: string } | null>(null);

  const checklist = externalChecklist || internalChecklist;

  const toggleCheck = (id: string) => {
    if (externalOnToggleCheck) {
      externalOnToggleCheck(id);
    } else {
      setInternalChecklist(prev =>
        prev.map(item => (item.id === id ? { ...item, checked: !item.checked } : item))
      );
    }

    const item = checklist.find(i => i.id === id);
    if (item) {
      const nextVal = !item.checked;
      setActionNotice({
        action: `${item.title} — ${nextVal ? 'Passed' : 'Pending'}`,
        detail: `Inspection item #${item.id} status updated in preview state. No telematics sync required.`
      });
    }
  };

  const handleGateClearance = () => {
    setActionNotice({
      action: 'Dock Gate Clearance Signed Off',
      detail: 'Gate Pass #GP-2026-BLR-8921 cleared for Departure at Hub Gate 3. Recorded in local preview state.'
    });
  };

  const checkedCount = checklist.filter(c => c.checked).length;
  const isAllChecked = checkedCount === checklist.length;

  return (
    <SafeAreaView style={styles.container}>
      {backendManifestActive ? (
        <View style={styles.backendNotice}>
          <DataBadge label="Backend manifest" tone="success" />
          <Text style={styles.backendTitle}>Cargo and dock checks are not connected</Text>
          <Text style={styles.backendDescription}>The active backend manifest contains route stops, but its API does not provide package details, cargo weights, seal IDs, inspection checklist state, or gate clearance. Those controls are hidden while paired so sample cargo can’t be mistaken for assigned freight.</Text>
        </View>
      ) : (
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>03 · LOAD MANIFEST & DOCK AUDIT</Text>
            <Text style={styles.title}>Cargo Manifest & Dock Audit</Text>
          </View>
          <DataBadge label="Preview" tone="preview" />
        </View>

        {/* Global Action Feedback Notice */}
        {actionNotice && (
          <SimulatedActionNotice
            actionName={actionNotice.action}
            detail={actionNotice.detail}
            onDismiss={() => setActionNotice(null)}
          />
        )}

        {/* Cargo Capacity Overview Card (Ref 4 Maer Warehouse Distribution) */}
        <View style={styles.capacityCard}>
          <View style={styles.capacityHeader}>
            <View>
              <Text style={styles.capacityLabel}>ACTIVE VAN PAYLOAD CAPACITY</Text>
              <Text style={styles.capacityValue}>
                {DRIVER_PROFILE.currentPayloadKg} / {DRIVER_PROFILE.maxPayloadKg} kg
              </Text>
            </View>
            <View style={styles.badgeCapacity}>
              <Text style={styles.badgeCapacityText}>
                {Math.round((DRIVER_PROFILE.currentPayloadKg / DRIVER_PROFILE.maxPayloadKg) * 100)}% Utilized
              </Text>
            </View>
          </View>

          {/* Segmented Cargo Type Bar */}
          <View style={styles.segmentContainer}>
            <View style={[styles.segmentBar, { flex: 43, backgroundColor: '#205A46' }]} />
            <View style={[styles.segmentBar, { flex: 33, backgroundColor: '#2F5BEA' }]} />
            <View style={[styles.segmentBar, { flex: 24, backgroundColor: '#D78A45' }]} />
          </View>
          <View style={styles.segmentLegend}>
            <Text style={styles.legendText}>■ Pharma (180kg / Cold 4°C)</Text>
            <Text style={styles.legendText}>■ Perishables (140kg)</Text>
            <Text style={styles.legendText}>■ Electronics (100kg)</Text>
          </View>
        </View>

        {/* Active Consignment Manifest List (Ref 6 Tipax) */}
        <View style={styles.sectionHeader}>
          <View>
            <Text style={styles.sectionTitle}>Manifest Consignments ({MANIFEST_CONSIGNMENTS.length})</Text>
            <Text style={styles.sectionSub}>Loaded cargo units assigned to Shift R-04</Text>
          </View>
          <Text style={styles.consignmentsTotalWeight}>Total: 420 kg</Text>
        </View>

        <View style={styles.manifestList}>
          {MANIFEST_CONSIGNMENTS.map(item => (
            <View key={item.trackingId} style={styles.manifestCard}>
              <View style={styles.manifestTop}>
                <View style={styles.manifestIdBadge}>
                  <Text style={styles.manifestIdText}>{item.trackingId}</Text>
                </View>
                <Text style={styles.manifestStopNum}>Stop #{item.stopSequence}</Text>
              </View>

              <Text style={styles.manifestCustName}>{item.customerName}</Text>

              <View style={styles.manifestMetaRow}>
                <Text style={styles.manifestMetaText}>{item.packages} packages · {item.weightKg} kg</Text>
                <Text style={styles.manifestMetaText}>· {item.category}</Text>
                {item.tempControlled && (
                  <Text style={styles.manifestColdText}>· Cold-chain</Text>
                )}
              </View>

              <View style={styles.sealRow}>
                <Text style={styles.sealLabel}>Tamper Seal:</Text>
                <Text style={styles.sealValue}>{item.sealNumber}</Text>
              </View>
            </View>
          ))}
        </View>

        {/* Dock Departure Checklist Section (Skills/driver-android-expo §9) */}
        <View style={[styles.sectionHeader, { marginTop: 22 }]}>
          <View>
            <Text style={styles.sectionTitle}>Dock Departure Checklist</Text>
            <Text style={styles.sectionSub}>Verify all 5 safety & cargo points before dispatch</Text>
          </View>
          <Text style={[styles.checkCounter, isAllChecked && styles.checkCounterDone]}>
            {checkedCount}/{checklist.length} Passed
          </Text>
        </View>

        <View style={styles.checklistWrap}>
          {checklist.map(item => (
            <TouchableOpacity
              key={item.id}
              style={[styles.checkItem, item.checked && styles.checkItemChecked]}
              onPress={() => toggleCheck(item.id)}
              activeOpacity={0.7}
              hitSlop={{ top: 4, bottom: 4, left: 4, right: 4 }}
            >
              <View style={[styles.checkbox, item.checked && styles.checkboxChecked]}>
                {item.checked && <Text style={styles.checkTick}>✓</Text>}
              </View>

              <View style={styles.itemTextWrap}>
                <View style={styles.itemTitleRow}>
                  <Text style={[styles.itemTitle, item.checked && styles.itemTitleChecked]}>
                    {item.title}
                  </Text>
                  <View style={styles.categoryPill}>
                    <Text style={styles.categoryPillText}>{item.category.toUpperCase()}</Text>
                  </View>
                </View>
                <Text style={styles.itemDescription}>{item.description}</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Departure Sign-off CTA */}
        <TouchableOpacity
          style={[styles.signoffBtn, !isAllChecked && styles.signoffBtnDisabled]}
          disabled={!isAllChecked}
          onPress={handleGateClearance}
          activeOpacity={0.8}
        >
          <Text style={styles.signoffBtnText}>
            {isAllChecked ? '✓ Authorize Dock Gate Clearance' : 'Complete All Inspection Checks (5/5 Required)'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
      )}
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
  backendNotice: { margin: 16, padding: 20, gap: 12, backgroundColor: '#FFFFFF', borderRadius: 18, borderWidth: 1, borderColor: 'rgba(21, 33, 29, 0.08)' },
  backendTitle: { fontSize: 18, fontWeight: '800', color: '#15211D' },
  backendDescription: { fontSize: 13, lineHeight: 20, color: '#68756F' },
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
  capacityCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
    marginBottom: 18,
  },
  capacityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  capacityLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: '#77736D',
    fontFamily: 'monospace',
  },
  capacityValue: {
    fontSize: 22,
    fontWeight: '800',
    color: '#15211D',
    marginTop: 2,
  },
  badgeCapacity: {
    backgroundColor: '#E2F4EE',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  badgeCapacityText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#205A46',
  },
  segmentContainer: {
    flexDirection: 'row',
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
    marginTop: 14,
    gap: 3,
  },
  segmentBar: {
    height: '100%',
    borderRadius: 2,
  },
  segmentLegend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 10,
  },
  legendText: {
    fontSize: 10,
    fontWeight: '600',
    color: '#55534F',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
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
  consignmentsTotalWeight: {
    fontSize: 11,
    fontWeight: '700',
    color: '#205A46',
  },
  manifestList: {
    gap: 10,
  },
  manifestCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
  },
  manifestTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  manifestIdBadge: {
    backgroundColor: '#F3F4F6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  manifestIdText: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: '700',
    color: '#205A46',
  },
  manifestStopNum: {
    fontSize: 10,
    fontWeight: '700',
    color: '#77736D',
  },
  manifestCustName: {
    fontSize: 13,
    fontWeight: '700',
    color: '#15211D',
    marginBottom: 4,
  },
  manifestMetaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
    alignItems: 'center',
  },
  manifestMetaText: {
    fontSize: 10,
    color: '#55534F',
  },
  manifestColdText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#0284C7',
  },
  sealRow: {
    flexDirection: 'row',
    marginTop: 6,
    paddingTop: 6,
    borderTopWidth: 1,
    borderTopColor: 'rgba(21, 33, 29, 0.04)',
    alignItems: 'center',
  },
  sealLabel: {
    fontSize: 10,
    color: '#77736D',
    marginRight: 4,
  },
  sealValue: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: '700',
    color: '#15211D',
  },
  checkCounter: {
    fontSize: 11,
    fontWeight: '700',
    color: '#77736D',
  },
  checkCounterDone: {
    color: '#205A46',
  },
  checklistWrap: {
    gap: 10,
  },
  checkItem: {
    backgroundColor: '#FFFFFF',
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(21, 33, 29, 0.08)',
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  checkItemChecked: {
    backgroundColor: '#FAFBF9',
    borderColor: '#A8DBC9',
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#929E99',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 10,
    marginTop: 2,
  },
  checkboxChecked: {
    backgroundColor: '#205A46',
    borderColor: '#205A46',
  },
  checkTick: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '900',
  },
  itemTextWrap: {
    flex: 1,
  },
  itemTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  itemTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#15211D',
    flex: 1,
    marginRight: 6,
  },
  itemTitleChecked: {
    color: '#205A46',
  },
  categoryPill: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: '#F3F4F6',
    borderRadius: 4,
  },
  categoryPillText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#4B5563',
  },
  itemDescription: {
    fontSize: 11,
    color: '#77736D',
    marginTop: 3,
    lineHeight: 16,
  },
  signoffBtn: {
    marginTop: 20,
    backgroundColor: '#17231F',
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  signoffBtnDisabled: {
    backgroundColor: '#55605B',
  },
  signoffBtnText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '800',
  },
});
