import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

interface SimulatedActionNoticeProps {
  actionName: string;
  detail: string;
  onDismiss: () => void;
  dark?: boolean;
}

export const SimulatedActionNotice: React.FC<SimulatedActionNoticeProps> = ({
  actionName,
  detail,
  onDismiss,
  dark = false,
}) => {
  return (
    <View style={[styles.container, dark ? styles.containerDark : styles.containerLight]}>
      <View style={styles.topRow}>
        <View style={styles.tagWrap}>
          <Text style={[styles.tag, dark ? styles.tagDark : styles.tagLight]}>
            [Simulated Action — No Backend Endpoint Connected]
          </Text>
        </View>
        <TouchableOpacity
          onPress={onDismiss}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
          accessibilityRole="button"
          accessibilityLabel="Dismiss simulated action notice"
        >
          <Text style={[styles.dismissText, dark ? styles.dismissDark : styles.dismissLight]}>✕</Text>
        </TouchableOpacity>
      </View>
      <Text style={[styles.actionName, dark ? styles.actionNameDark : styles.actionNameLight]}>
        {actionName}
      </Text>
      <Text style={[styles.detail, dark ? styles.detailDark : styles.detailLight]}>
        {detail}
      </Text>
      <Text style={[styles.footnote, dark ? styles.footnoteDark : styles.footnoteLight]}>
        Hard Rule 7 · arQ driver endpoints deferred to Phase D. Recorded in local preview state.
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    marginVertical: 10,
  },
  containerLight: {
    backgroundColor: '#FDF8F0',
    borderColor: '#FBD6B0',
  },
  containerDark: {
    backgroundColor: 'rgba(215, 138, 69, 0.16)',
    borderColor: 'rgba(215, 138, 69, 0.45)',
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  tagWrap: {
    flex: 1,
    marginRight: 8,
  },
  tag: {
    fontSize: 10,
    fontFamily: 'monospace',
    fontWeight: '800',
    letterSpacing: 0.2,
  },
  tagLight: {
    color: '#B26B00',
  },
  tagDark: {
    color: '#FBD6B0',
  },
  dismissText: {
    fontSize: 15,
    fontWeight: '700',
    paddingHorizontal: 4,
  },
  dismissLight: {
    color: '#4A5565',
  },
  dismissDark: {
    color: '#D9E6DC',
  },
  actionName: {
    fontSize: 13,
    fontWeight: '800',
    marginTop: 2,
  },
  actionNameLight: {
    color: '#0F1720',
  },
  actionNameDark: {
    color: '#FFFFFF',
  },
  detail: {
    fontSize: 12,
    marginTop: 2,
    lineHeight: 16,
  },
  detailLight: {
    color: '#4A5565',
  },
  detailDark: {
    color: '#C7D4CE',
  },
  footnote: {
    fontSize: 10,
    marginTop: 6,
    fontStyle: 'italic',
  },
  footnoteLight: {
    color: '#8A7A66',
  },
  footnoteDark: {
    color: '#A0B2AA',
  },
});
