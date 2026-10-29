export interface DriverStop {
  id: string;
  sequence: number;
  customerName: string;
  consignee: string;
  address: string;
  trackingId: string;
  packageCount: number;
  windowStart: string;
  windowEnd: string;
  etaInterval: string;
  pLate: string;
  demandKg: number;
  status: 'completed' | 'in_transit' | 'pending' | 'at_risk';
  isLocked: boolean;
  notes?: string;
  corridorAdvisory?: string;
  contactName: string;
  contactPhone: string;
  contactRole: string;
  specialHandling: string;
  maneuverInstruction: string;
  maneuverDistance: string;
  maneuverNext: string;
  maneuverIcon: string;
}

export interface DockChecklistItem {
  id: string;
  title: string;
  description: string;
  checked: boolean;
  category: 'safety' | 'cargo' | 'document';
}

export interface ManifestConsignment {
  trackingId: string;
  stopSequence: number;
  customerName: string;
  packages: number;
  weightKg: number;
  category: string;
  tempControlled: boolean;
  sealNumber: string;
}

export interface ShiftSchedule {
  shiftId: string;
  shiftHours: string;
  earningsInr: number;
  stops: DriverStop[];
}

export const DRIVER_PROFILE = {
  driverId: 'DRV-BLR-084',
  name: 'Ramesh Gowda',
  phone: '+91 98450 12345',
  vehicleCallsign: 'EV-ACE-01',
  vehicleModel: 'Tata Ace EV Pro (2026)',
  licenseNumber: 'KA-01-2022-009841',
  rating: 4.92,
  completedTrips: 1420,
  batteryPct: 82,
  currentPayloadKg: 420,
  maxPayloadKg: 500,
  speedKmh: 34,
  speedLimitKmh: 40,
  locationLabel: 'Indiranagar 100ft Rd, Bengaluru',
  todayEarningsInr: 2450,
  assumedRateLabel: 'assumed rate ₹83/$',
  shiftId: 'Shift R-04',
  shiftHours: '13:30 – 21:30 IST'
};

export const SHIFT_DATES = [
  { day: 'Wed', date: '29 Oct', shift: 'Shift R-04', stops: 4, active: true },
  { day: 'Thu', date: '30 Oct', shift: 'Shift R-08', stops: 5, active: false },
  { day: 'Fri', date: '31 Oct', shift: 'Shift R-02', stops: 4, active: false },
  { day: 'Sat', date: '01 Nov', shift: 'Shift R-09', stops: 3, active: false }
];

export const DRIVER_STOPS: DriverStop[] = [
  {
    id: 'stp-01',
    sequence: 1,
    customerName: 'HealthKart Fulfilment Hub',
    consignee: 'HealthKart Warehouse 4B',
    address: 'Indiranagar 12th Main, Bengaluru',
    trackingId: 'TRK-BLR-8921-A',
    packageCount: 3,
    windowStart: '14:00',
    windowEnd: '14:30',
    etaInterval: '14:10 – 14:16 IST (simulated)',
    pLate: '0% (simulated)',
    demandKg: 85,
    status: 'completed',
    isLocked: true,
    contactName: 'Anil Kumar',
    contactPhone: '+91 98765 11001',
    contactRole: 'Inbound Dock Supervisor',
    specialHandling: 'Cold-Chain 4.0°C Verified',
    maneuverInstruction: 'Depart Hub via Gate 3 onto 12th Main',
    maneuverDistance: 'In 100 meters',
    maneuverNext: 'Head south toward Indiranagar 100ft Rd',
    maneuverIcon: '↑'
  },
  {
    id: 'stp-02',
    sequence: 2,
    customerName: 'Apollo Pharmacy Regional Hub',
    consignee: 'Apollo Central Distribution',
    address: 'Koramangala 4th Block, Bengaluru',
    trackingId: 'TRK-BLR-8922-B',
    packageCount: 4,
    windowStart: '14:30',
    windowEnd: '15:00',
    etaInterval: '14:38 – 14:46 IST (simulated)',
    pLate: '6% (simulated)',
    demandKg: 120,
    status: 'in_transit',
    isLocked: false,
    notes: 'Temperature-controlled cargo: keep sealed at 4°C',
    corridorAdvisory: 'Approaching Sony World junction: slight congestion (+4 min)',
    contactName: 'Priya Sharma',
    contactPhone: '+91 98765 11002',
    contactRole: 'Receiving Pharmacist',
    specialHandling: 'Cold-Chain 4.2°C (2°C – 8°C range)',
    maneuverInstruction: 'Turn right onto 100 Feet Road',
    maneuverDistance: 'In 350 meters',
    maneuverNext: 'Then in 1.4 km merge onto Koramangala 80ft Rd',
    maneuverIcon: '↱'
  },
  {
    id: 'stp-03',
    sequence: 3,
    customerName: 'BigBasket Distribution Center',
    consignee: 'BigBasket Dark Store #14',
    address: 'HSR Layout Sector 1, Bengaluru',
    trackingId: 'TRK-BLR-8923-C',
    packageCount: 6,
    windowStart: '15:15',
    windowEnd: '15:45',
    etaInterval: '15:28 – 15:36 IST (simulated)',
    pLate: '9% (simulated)',
    demandKg: 110,
    status: 'pending',
    isLocked: false,
    contactName: 'Suresh Babu',
    contactPhone: '+91 98765 11003',
    contactRole: 'Operations Lead',
    specialHandling: 'Perishables / Ambient Pallet',
    maneuverInstruction: 'Continue onto Hosur Rd toward HSR Layout',
    maneuverDistance: 'In 1.2 km',
    maneuverNext: 'Take 14th Main exit toward Sector 1',
    maneuverIcon: '↑'
  },
  {
    id: 'stp-04',
    sequence: 4,
    customerName: 'Croma Logistics Point',
    consignee: 'Croma Regional Hub Bay 2',
    address: 'Bellandur EcoSpace, Bengaluru',
    trackingId: 'TRK-BLR-8924-D',
    packageCount: 2,
    windowStart: '16:00',
    windowEnd: '16:45',
    etaInterval: '16:32 – 16:44 IST (simulated)',
    pLate: '38% (simulated)',
    demandKg: 105,
    status: 'at_risk',
    isLocked: false,
    notes: 'Corridor congestion: Outer Ring Road peak surge recommended for re-routing',
    corridorAdvisory: 'Advisory: +12 min Corridor Delay on Outer Ring Road',
    contactName: 'Deepak Rao',
    contactPhone: '+91 98765 11004',
    contactRole: 'Gate Security & Inward Officer',
    specialHandling: 'High-Value Consumer Electronics',
    maneuverInstruction: 'Take Bellandur Flyover exit onto Outer Ring Rd',
    maneuverDistance: 'In 2.8 km',
    maneuverNext: 'Arrive at EcoSpace Main Gate Inward Bay',
    maneuverIcon: '↱'
  }
];

export const SHIFT_SCHEDULES: Record<string, ShiftSchedule> = {
  '29 Oct': {
    shiftId: 'Shift R-04',
    shiftHours: '13:30 – 21:30 IST',
    earningsInr: 2450,
    stops: DRIVER_STOPS
  },
  '30 Oct': {
    shiftId: 'Shift R-08',
    shiftHours: '09:00 – 17:00 IST',
    earningsInr: 2890,
    stops: [
      {
        id: 'stp-30-01',
        sequence: 1,
        customerName: 'MedPlus Pharma Central Depot',
        consignee: 'Depot Intake Dock 2',
        address: 'Whitefield Main Rd, Bengaluru',
        trackingId: 'TRK-BLR-9011-A',
        packageCount: 5,
        windowStart: '09:30',
        windowEnd: '10:00',
        etaInterval: '09:40 – 09:48 IST (simulated)',
        pLate: '0% (simulated)',
        demandKg: 95,
        status: 'pending',
        isLocked: true,
        contactName: 'Kavitha N',
        contactPhone: '+91 98765 22001',
        contactRole: 'Inventory Manager',
        specialHandling: 'Cold-Chain 4.0°C Verified',
        maneuverInstruction: 'Merge onto Whitefield Main Road',
        maneuverDistance: 'In 500 meters',
        maneuverNext: 'Continue past Hope Farm junction',
        maneuverIcon: '↑'
      },
      {
        id: 'stp-30-02',
        sequence: 2,
        customerName: 'Zepto Fulfillment Hub #08',
        consignee: 'Inward Staging Area',
        address: 'ITPL Tech Park, Bengaluru',
        trackingId: 'TRK-BLR-9012-B',
        packageCount: 8,
        windowStart: '10:30',
        windowEnd: '11:00',
        etaInterval: '10:42 – 10:50 IST (simulated)',
        pLate: '4% (simulated)',
        demandKg: 130,
        status: 'pending',
        isLocked: false,
        contactName: 'Rajesh Nair',
        contactPhone: '+91 98765 22002',
        contactRole: 'Dock Coordinator',
        specialHandling: 'Quick-Commerce Rapid Handover',
        maneuverInstruction: 'Turn left toward ITPL Gate 2',
        maneuverDistance: 'In 800 meters',
        maneuverNext: 'Enter commercial vehicle lane',
        maneuverIcon: '↰'
      },
      {
        id: 'stp-30-03',
        sequence: 3,
        customerName: 'Blinkit Staging Hub #19',
        consignee: 'Bay 4 Distribution',
        address: 'Hoodi Circle, Bengaluru',
        trackingId: 'TRK-BLR-9013-C',
        packageCount: 4,
        windowStart: '11:45',
        windowEnd: '12:15',
        etaInterval: '11:55 – 12:05 IST (simulated)',
        pLate: '8% (simulated)',
        demandKg: 75,
        status: 'pending',
        isLocked: false,
        contactName: 'Vinod Hegde',
        contactPhone: '+91 98765 22003',
        contactRole: 'Hub Lead',
        specialHandling: 'Perishables',
        maneuverInstruction: 'Roundabout 2nd exit toward Hoodi',
        maneuverDistance: 'In 1.5 km',
        maneuverNext: 'Merge onto Outer Ring road link',
        maneuverIcon: '↑'
      },
      {
        id: 'stp-30-04',
        sequence: 4,
        customerName: 'Reliance Digital Supply Bay',
        consignee: 'High-Value Vault 1',
        address: 'Mahadevapura, Bengaluru',
        trackingId: 'TRK-BLR-9014-D',
        packageCount: 3,
        windowStart: '13:00',
        windowEnd: '13:45',
        etaInterval: '13:15 – 13:25 IST (simulated)',
        pLate: '12% (simulated)',
        demandKg: 90,
        status: 'pending',
        isLocked: false,
        contactName: 'Manoj Kumar',
        contactPhone: '+91 98765 22004',
        contactRole: 'Security Supervisor',
        specialHandling: 'Secure Tamper Latch Required',
        maneuverInstruction: 'Turn right at Mahadevapura flyover',
        maneuverDistance: 'In 2.1 km',
        maneuverNext: 'Proceed to commercial loading ramp',
        maneuverIcon: '↱'
      },
      {
        id: 'stp-30-05',
        sequence: 5,
        customerName: 'Amazon Sortation Center BLR-4',
        consignee: 'Crossdock Intake 6',
        address: 'KR Puram Ring Rd, Bengaluru',
        trackingId: 'TRK-BLR-9015-E',
        packageCount: 6,
        windowStart: '14:30',
        windowEnd: '15:15',
        etaInterval: '14:45 – 14:58 IST (simulated)',
        pLate: '22% (simulated)',
        demandKg: 110,
        status: 'pending',
        isLocked: false,
        contactName: 'Geetha Swamy',
        contactPhone: '+91 98765 22005',
        contactRole: 'Crossdock Lead',
        specialHandling: 'Sortation Pallets',
        maneuverInstruction: 'Take KR Puram bypass ramp',
        maneuverDistance: 'In 3.4 km',
        maneuverNext: 'Approach security booth 4',
        maneuverIcon: '↑'
      }
    ]
  },
  '31 Oct': {
    shiftId: 'Shift R-02',
    shiftHours: '06:00 – 14:00 IST',
    earningsInr: 2200,
    stops: [
      {
        id: 'stp-31-01',
        sequence: 1,
        customerName: 'FreshToHome Cold Facility',
        consignee: 'Temperature Chamber A',
        address: 'Yeshwanthpur, Bengaluru',
        trackingId: 'TRK-BLR-9101-A',
        packageCount: 6,
        windowStart: '06:45',
        windowEnd: '07:15',
        etaInterval: '06:55 – 07:05 IST (simulated)',
        pLate: '0% (simulated)',
        demandKg: 140,
        status: 'pending',
        isLocked: true,
        contactName: 'Sanjay Reddy',
        contactPhone: '+91 98765 33001',
        contactRole: 'Cold Store Head',
        specialHandling: 'Deep Chill -2°C to +2°C',
        maneuverInstruction: 'North on Tumkur Road',
        maneuverDistance: 'In 1.8 km',
        maneuverNext: 'Take Yeshwanthpur APMC exit',
        maneuverIcon: '↑'
      },
      {
        id: 'stp-31-02',
        sequence: 2,
        customerName: 'Licious Fulfillment Station',
        consignee: 'Dock Bay 3',
        address: 'Malleshwaram 8th Cross, Bengaluru',
        trackingId: 'TRK-BLR-9102-B',
        packageCount: 4,
        windowStart: '07:45',
        windowEnd: '08:15',
        etaInterval: '07:55 – 08:05 IST (simulated)',
        pLate: '5% (simulated)',
        demandKg: 85,
        status: 'pending',
        isLocked: false,
        contactName: 'Arjun Das',
        contactPhone: '+91 98765 33002',
        contactRole: 'Receiving In-charge',
        specialHandling: 'Chilled Meat Transit',
        maneuverInstruction: 'Turn left onto Margosa Road',
        maneuverDistance: 'In 900 meters',
        maneuverNext: 'Arrive at Malleshwaram 8th Cross',
        maneuverIcon: '↰'
      },
      {
        id: 'stp-31-03',
        sequence: 3,
        customerName: 'Nature Basket Hub',
        consignee: 'Store Intake',
        address: 'Rajajinagar 1st Block, Bengaluru',
        trackingId: 'TRK-BLR-9103-C',
        packageCount: 3,
        windowStart: '08:45',
        windowEnd: '09:15',
        etaInterval: '08:52 – 09:02 IST (simulated)',
        pLate: '7% (simulated)',
        demandKg: 70,
        status: 'pending',
        isLocked: false,
        contactName: 'Fatima Z',
        contactPhone: '+91 98765 33003',
        contactRole: 'Store Manager',
        specialHandling: 'Organic Groceries',
        maneuverInstruction: 'Merge onto Dr Rajkumar Road',
        maneuverDistance: 'In 1.1 km',
        maneuverNext: 'Turn into commercial driveway',
        maneuverIcon: '↱'
      },
      {
        id: 'stp-31-04',
        sequence: 4,
        customerName: 'Spar Hypermarket Distribution',
        consignee: 'Central Loading Dock',
        address: 'Nagavara Junction, Bengaluru',
        trackingId: 'TRK-BLR-9104-D',
        packageCount: 5,
        windowStart: '10:00',
        windowEnd: '10:45',
        etaInterval: '10:12 – 10:24 IST (simulated)',
        pLate: '18% (simulated)',
        demandKg: 125,
        status: 'pending',
        isLocked: false,
        contactName: 'Chandan Roy',
        contactPhone: '+91 98765 33004',
        contactRole: 'Logistics Supervisor',
        specialHandling: 'FMCG Bulk Crates',
        maneuverInstruction: 'Take Outer Ring Rd towards Hebbal',
        maneuverDistance: 'In 4.2 km',
        maneuverNext: 'Arrive at Spar Mega Ramp',
        maneuverIcon: '↑'
      }
    ]
  },
  '01 Nov': {
    shiftId: 'Shift R-09',
    shiftHours: '14:00 – 21:00 IST',
    earningsInr: 2100,
    stops: [
      {
        id: 'stp-01-01',
        sequence: 1,
        customerName: 'Tata 1mg Regional Center',
        consignee: 'Inbound Warehouse B',
        address: 'Peenya 2nd Stage, Bengaluru',
        trackingId: 'TRK-BLR-9201-A',
        packageCount: 7,
        windowStart: '14:30',
        windowEnd: '15:15',
        etaInterval: '14:45 – 14:55 IST (simulated)',
        pLate: '0% (simulated)',
        demandKg: 160,
        status: 'pending',
        isLocked: true,
        contactName: 'Harish Babu',
        contactPhone: '+91 98765 44001',
        contactRole: 'Quality Assurance Head',
        specialHandling: 'Pharmaceutical Cold-Chain',
        maneuverInstruction: 'West on Tumkur Road industrial corridor',
        maneuverDistance: 'In 3.0 km',
        maneuverNext: 'Turn right at Peenya 2nd Stage entrance',
        maneuverIcon: '↱'
      },
      {
        id: 'stp-01-02',
        sequence: 2,
        customerName: 'Schneider Electric Logistics',
        consignee: 'Industrial Staging Bay 3',
        address: 'Jalahalli Cross, Bengaluru',
        trackingId: 'TRK-BLR-9202-B',
        packageCount: 4,
        windowStart: '15:45',
        windowEnd: '16:30',
        etaInterval: '16:00 – 16:12 IST (simulated)',
        pLate: '10% (simulated)',
        demandKg: 110,
        status: 'pending',
        isLocked: false,
        contactName: 'Naveen Kumar',
        contactPhone: '+91 98765 44002',
        contactRole: 'Plant Inward Lead',
        specialHandling: 'Industrial Automation Sensors',
        maneuverInstruction: 'Continue toward Jalahalli metro',
        maneuverDistance: 'In 2.2 km',
        maneuverNext: 'Turn left into plant security gate',
        maneuverIcon: '↰'
      },
      {
        id: 'stp-01-03',
        sequence: 3,
        customerName: 'Flipkart Supermart Station',
        consignee: 'Perishables Intake Hub',
        address: 'Yelahanka New Town, Bengaluru',
        trackingId: 'TRK-BLR-9203-C',
        packageCount: 5,
        windowStart: '17:15',
        windowEnd: '18:00',
        etaInterval: '17:30 – 17:42 IST (simulated)',
        pLate: '15% (simulated)',
        demandKg: 130,
        status: 'pending',
        isLocked: false,
        contactName: 'Sunita M',
        contactPhone: '+91 98765 44003',
        contactRole: 'Operations Manager',
        specialHandling: 'Chilled Milk & Fresh Produce',
        maneuverInstruction: 'Take airport road toward Yelahanka',
        maneuverDistance: 'In 5.5 km',
        maneuverNext: 'Enter Supermart dispatch alley',
        maneuverIcon: '↑'
      }
    ]
  }
};

export const MANIFEST_CONSIGNMENTS: ManifestConsignment[] = [
  {
    trackingId: 'TRK-BLR-8921-A',
    stopSequence: 1,
    customerName: 'HealthKart Fulfilment Hub',
    packages: 3,
    weightKg: 85,
    category: 'Nutraceuticals',
    tempControlled: true,
    sealNumber: 'SEC-9941'
  },
  {
    trackingId: 'TRK-BLR-8922-B',
    stopSequence: 2,
    customerName: 'Apollo Pharmacy Regional Hub',
    packages: 4,
    weightKg: 120,
    category: 'Critical Pharmaceuticals',
    tempControlled: true,
    sealNumber: 'SEC-9942'
  },
  {
    trackingId: 'TRK-BLR-8923-C',
    stopSequence: 3,
    customerName: 'BigBasket Distribution Center',
    packages: 6,
    weightKg: 110,
    category: 'Perishable Groceries',
    tempControlled: false,
    sealNumber: 'SEC-9943'
  },
  {
    trackingId: 'TRK-BLR-8924-D',
    stopSequence: 4,
    customerName: 'Croma Logistics Point',
    packages: 2,
    weightKg: 105,
    category: 'High-Value Electronics',
    tempControlled: false,
    sealNumber: 'SEC-9944'
  }
];

export const DOCK_CHECKLIST: DockChecklistItem[] = [
  {
    id: 'chk-1',
    title: 'Cold-Chain Chamber Temperature Audit',
    description: 'Chamber temperature locked at 4.2°C (permitted range: 2.0°C – 8.0°C). Sensor calibrated.',
    checked: true,
    category: 'cargo'
  },
  {
    id: 'chk-2',
    title: 'Pallet Ratchet Tie-Down Tension',
    description: 'Cargo tie-down straps tensioned to 250 daN across all 4 pallet bays.',
    checked: true,
    category: 'safety'
  },
  {
    id: 'chk-3',
    title: 'Electronic Gate Pass & Consignment Manifest',
    description: 'Digital Pass #GP-2026-BLR-8921 scanned and validated at Main Hub Security Gate 3.',
    checked: true,
    category: 'document'
  },
  {
    id: 'chk-4',
    title: 'EV Battery Reserve & Range Buffer',
    description: '82% SoC (148 km estimated range) exceeds required 60% threshold for scheduled route.',
    checked: true,
    category: 'safety'
  },
  {
    id: 'chk-5',
    title: 'Tamper-Evident Rear Roller Seal',
    description: 'High-security barrier seal #SEC-9941 inspected and locked over rear roller door latch.',
    checked: false,
    category: 'cargo'
  }
];
