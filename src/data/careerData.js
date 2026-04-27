export const CAREER_ARTICLES = [
  {
    id: "1",
    type: "overview",
    title: "Careers & Color Vision",
    tag: "Career",
    readTime: "3 min read",
    intro:
      "Some professions require normal color perception. Knowing which ones helps you make confident, informed career choices.",
    // Overview specific layout
    cards: [
      {
        label: "Aviation",
        desc: "Navigation lights and weather radar require color discrimination",
        icon: "airplane-outline",
        color: "#FEF2F2",
      },
      {
        label: "Electrical",
        desc: "Wire color codes are mandatory safety standards",
        icon: "flash-outline",
        color: "#FFFBEB",
      },
      {
        label: "Healthcare",
        desc: "Tissue, blood & specimen color assessment in clinical settings",
        icon: "heart-outline",
        color: "#F5F3FF",
      },
    ],
    facts: [
      "Color vision requirements vary by profession and region",
      "CVD does not restrict the majority of occupations",
      "Early awareness enables better career planning",
      "CVD-friendly alternatives exist in every affected field",
    ],
    roles: [],
  },
  {
    id: "2",
    title: "Piloting & Aviation",
    tag: "Aviation",
    readTime: "5 min read",
    intro:
      "Pilots must identify navigation lights, signal flares, and weather radar. Mandatory color vision tests apply globally via FAA and ICAO standards.",
    stats: [
      {
        label: "Required test",
        value: "FAA",
        color: "#FEE2E2",
        textColor: "#EF4444",
      },
      {
        label: "Global standard",
        value: "ICAO",
        color: "#EDE9FE",
        textColor: "#8B5CF6",
      },
      {
        label: "CVD limitations",
        value: "3+",
        color: "#E0F2FE",
        textColor: "#0EA5E9",
      },
    ],
    sections: [
      {
        title: "Why Color Vision Matters",
        type: "list",
        items: [
          {
            label: "Port Navigation Light",
            desc: "Red — marks the left (port) side",
            hex: "#EF4444",
          },
          {
            label: "Starboard Navigation Light",
            desc: "Green — marks the right (starboard) side",
            hex: "#22C55E",
          },
          {
            label: "Weather Radar Color Scale",
            desc: "Color intensity indicates storm severity",
            hex: "#6B7280",
          },
        ],
      },
    ],
    roles: [
      "Aviation Meteorologist",
      "Ground Operations",
      "Drone Technician",
      "Flight Data Analyst",
    ],
  },
  {
    id: "3",
    title: "Electrical Work",
    tag: "Trades",
    readTime: "5 min read",
    intro:
      "Electrical wiring uses standardised color codes enforced by safety law. Misidentifying wire colors can cause shock, fire, or fatality.",
    sections: [
      {
        title: "Live Wire (Brown / Red)",
        subtitle: "Highest risk",
        colors: ["#B91C1C", "#F87171", "#92400E", "#991B1B"],
        bgColor: "#FEF2F2",
        footer: "Confusion with neutral wire can cause lethal shock",
      },
      {
        title: "Neutral Wire (Blue)",
        subtitle: "Moderate risk",
        colors: ["#3B82F6", "#60A5FA", "#1D4ED8", "#22D3EE"],
        bgColor: "#EFF6FF",
        footer: "Must be clearly distinguished from live/earth wires",
      },
      {
        title: "Earth Wire (Green-Yellow)",
        subtitle: "Safety grounding",
        colors: ["#22C55E", "#FBBF24", "#15803D", "#A3E635"],
        bgColor: "#F0FDF4",
        footer: "Stripe pattern aids CVD workers but is still a challenge",
      },
    ],
    roles: [
      "Electrical Estimator",
      "CAD Designer",
      "Safety Inspector",
      "Building Auditor",
    ],
  },
  {
    id: "4",
    title: "Medical Professions",
    tag: "Healthcare",
    readTime: "5 min read",
    intro:
      "Certain medical specialties rely on color for accurate diagnosis. CVD awareness helps you choose suitable specialties.",
    sections: [
      {
        title: "Surgery",
        subtitle: "Red-weak vision (Protanomaly)",
        colors: ["#EF4444", "#F87171", "#991B1B", "#EC4899"],
        bgColor: "#FEF2F2",
        footer: "Blood color and tissue assessment are critical",
      },
      {
        title: "Dermatology",
        subtitle: "Skin lesion identification",
        colors: ["#F97316", "#F87171", "#B45309", "#78350F"],
        bgColor: "#FFFBEB",
        footer: "Diagnosing melanoma depends on hue recognition",
      },
      {
        title: "Pathology",
        subtitle: "Stained specimen analysis",
        colors: ["#8B5CF6", "#C4B5FD", "#EC4899", "#7C3AED"],
        bgColor: "#F5F3FF",
        footer: "Interpreting stained tissue samples under a microscope",
      },
    ],
    roles: [
      "Radiology",
      "Dentistry",
      "Medical Research",
      "Psychiatry",
      "Admin",
    ],
  },
  {
    id: "5",
    title: "Sports & Athletics",
    tag: "Sports",
    readTime: "4 min read",
    intro:
      "Athletes and officials must interpret color-coded signals quickly. CVD can affect performance and officiating.",
    sections: [
      {
        title: "Color-Critical Scenarios",
        type: "list",
        items: [
          {
            label: "Referee Signals",
            desc: "Red & yellow cards demand instant reading",
            hex: "#EF4444",
          },
          {
            label: "Team Jersey Identification",
            desc: "Fast-paced games rely on distinct jerseys",
            hex: "#3B82F6",
          },
          {
            label: "Line Markings",
            desc: "Boundary lines use contrasting color pairs",
            hex: "#F59E0B",
          },
        ],
      },
    ],
    facts: [
      "Most recreational sports remain accessible",
      "Professional refereeing may require clearance",
      "Sports tech is adopting CVD-safe designs",
    ],
    roles: [
      "Sports Commentator",
      "Athletic Coach",
      "Performance Analyst",
      "Team Manager",
    ],
  },
  {
    id: "6",
    title: "Traffic & Road Safety",
    tag: "Transport",
    readTime: "3 min read",
    intro:
      "Traffic signals and road signs rely on red-amber-green. Most countries allow CVD drivers via positional cues.",
    stats: [
      {
        label: "Top position",
        value: "Stop",
        color: "#FEF2F2",
        textColor: "#EF4444",
      },
      {
        label: "Middle position",
        value: "Caution",
        color: "#FFFBEB",
        textColor: "#F59E0B",
      },
      {
        label: "Bottom position",
        value: "Go",
        color: "#F0FDF4",
        textColor: "#10B981",
      },
    ],
    sections: [
      {
        title: "CVD Adaptations in Use",
        type: "list",
        items: [
          {
            label: "Position-Coded Signals",
            desc: "Top=Stop, Middle=Caution, Bottom=Go",
            hex: "#374151",
          },
          {
            label: "Symbol-Based Signs",
            desc: "Shape/text supplement color",
            hex: "#4B5563",
          },
        ],
      },
    ],
    roles: ["Road Planner", "Traffic Analyst", "Logistics Coordinator"],
  },
  {
    id: "7",
    title: "Art & Creative Design",
    tag: "Creative",
    readTime: "5 min read",
    intro:
      "Creative professions rely on color theory, but CVD hasn’t stopped celebrated artists.",
    stats: [
      {
        label: "Digital / RGB",
        value: "Screen",
        color: "#EFF6FF",
        textColor: "#3B82F6",
      },
      {
        label: "Print / CMYK",
        value: "Ink-based",
        color: "#FEF2F2",
        textColor: "#EC4899",
      },
    ],
    sections: [
      {
        title: "Color Accessibility in Design",
        type: "list",
        items: [
          {
            label: "WCAG Contrast",
            desc: "Minimum 4.5:1 ratio ensures readability",
            hex: "#8B5CF6",
          },
          {
            label: "Simulation Tools",
            desc: "Figma and Adobe include built-in CVD modes",
            hex: "#EC4899",
          },
        ],
      },
    ],
    roles: ["Typographer", "3D Modeler", "UX Researcher", "Motion Designer"],
  },
  {
    id: "8",
    title: "Food & Culinary Arts",
    tag: "Culinary",
    readTime: "4 min read",
    intro:
      "Color is a primary indicator of freshness, cooking completion, and food safety.",
    sections: [
      {
        title: "Freshness Assessment",
        subtitle: "Green indicates optimal ripeness",
        colors: ["#22C55E", "#A3E635", "#15803D"],
        bgColor: "#F0FDF4",
        footer: "Green produce turns yellow or brown as it ages",
      },
      {
        title: "Cooking Completion",
        subtitle: "Color changes signal doneness",
        colors: ["#B91C1C", "#F97316", "#451A03"],
        bgColor: "#FEF2F2",
        footer: "Meat transitions from red to brown when cooked",
      },
    ],
    roles: [
      "Food Technologist",
      "Menu Designer",
      "Kitchen Manager",
      "Nutritionist",
    ],
  },
  {
    id: "9",
    title: "Technology & UI Design",
    tag: "Technology",
    readTime: "4 min read",
    intro:
      "Digital interfaces use color to convey state. Building CVD-accessible products benefits all 300M+ users.",
    stats: [
      {
        label: "A11y standard",
        value: "WCAG",
        color: "#EFF6FF",
        textColor: "#3B82F6",
      },
      {
        label: "Min contrast",
        value: "4.5:1",
        color: "#F5F3FF",
        textColor: "#8B5CF6",
      },
      {
        label: "CVD users",
        value: "300M+",
        color: "#F0FDF4",
        textColor: "#10B981",
      },
    ],
    sections: [
      {
        title: "Color in UI Systems",
        type: "list",
        items: [
          {
            label: "Error & Alert States",
            desc: "Red error badges require icon backup",
            hex: "#EF4444",
          },
          {
            label: "Success Indicators",
            desc: "Green checkmarks must pair with text",
            hex: "#22C55E",
          },
        ],
      },
    ],
    roles: ["Frontend Developer", "Accessibility Auditor", "Product Manager"],
  },
];
