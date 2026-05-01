# ReColor — Chapter 3 Testing Methodology PlantUML Diagrams

> Six PlantUML diagrams for the Testing Methodology section of Chapter 3.
> Copy each `@startuml ... @enduml` block into a `.puml` file or PlantUML renderer.

---

## Diagram 1: Testing Framework Overview (Activity Diagram)

Shows the overall testing methodology pipeline from test planning through execution to reporting.

```plantuml
@startuml testing_framework_overview
!theme plain
skinparam backgroundColor #FFFFFF
skinparam ActivityBackgroundColor #E8EAF6
skinparam ActivityBorderColor #1F3864
skinparam ActivityFontSize 11
skinparam ArrowColor #1F3864
skinparam PartitionBackgroundColor #F5F5F5
skinparam PartitionBorderColor #1F3864
skinparam NoteBackgroundColor #FFF8E1
skinparam NoteBorderColor #F9A825

title **ReColor Testing Methodology Pipeline**

start

partition "Phase 1: Test Planning" {
  :Identify system modules\nand screen inventory;
  :Map functional requirements\nto test case categories;
  :Define test case ID schema\n(RC-*, FN-*, VE/AF/PS/SC-*, CM-*);
}

partition "Phase 2: Test Case Design" {
  fork
    :Part A\n**Module Tests**\n233 TCs\n(RC-* IDs);
    note right
      Authentication, Navigation,
      Camera, Correction, Simulation,
      Identifier, Ishihara, Haptic,
      Audio, Career, Settings,
      Firebase, Gallery
    end note
  fork again
    :Part B\n**Functionality Tests**\n107 TCs\n(FN-* IDs);
    note right
      Screen-by-screen
      feature verification
      (16 screen groups)
    end note
  fork again
    :Part C\n**Android Core Tests**\n80 TCs\n(VE/AF/PS/SC-* IDs);
    note right
      Visual Experience,
      Android Functionality,
      Performance & Stability,
      Privacy & Security
    end note
  fork again
    :Part D\n**Compatibility Tests**\n37 TCs\n(CM-* IDs);
    note right
      Android versions,
      Screen sizes, Camera HW,
      Network, Edge cases
    end note
  end fork
}

:Consolidate into\n**457 total test cases**;

partition "Phase 3: Test Execution" {
  :Deploy APK to\ntest devices;
  :Execute test cases\nmanually per checklist;
  :Record Pass / Fail\nwith remarks;
}

partition "Phase 4: Reporting" {
  :Compile results into\ntest summary report;
  :Perform coverage analysis\n(screen, component, quality dimension);
  :Identify gaps and\nrecommendations;
}

stop

@enduml
```

**Suggested Caption:** Figure 3.X: Testing Methodology Pipeline

---

## Diagram 2: Test Case Classification Hierarchy (Mind Map)

Shows how the 457 test cases are organized across the four document parts.

```plantuml
@startuml test_case_hierarchy
!theme plain
skinparam backgroundColor #FFFFFF
skinparam DefaultFontSize 11

<style>
mindmapDiagram {
  node {
    BackgroundColor #E8EAF6
    BorderColor #1F3864
    FontColor #1F3864
    RoundCorner 10
  }
  :depth(0) {
    BackgroundColor #1F3864
    FontColor #FFFFFF
    FontSize 14
    FontStyle bold
  }
  :depth(1) {
    BackgroundColor #2E74B5
    FontColor #FFFFFF
    FontSize 12
    FontStyle bold
  }
  :depth(2) {
    BackgroundColor #D6E4F0
    FontColor #1F3864
    FontSize 10
  }
}
</style>

title **ReColor Test Case Classification**

* 457 Test Cases
** Part A: Module Tests (233)
*** RC-AU: Authentication (20)
*** RC-NAV: Navigation (12)
*** RC-CAM: Camera (39)
*** RC-CC: Color Correction (27)
*** RC-SIM: CVD Simulation (7)
*** RC-CI: Color Identifier (20)
*** RC-ISH: Ishihara Screening (47)
*** RC-HF: Haptic Feedback (9)
*** RC-AF: Audio Feedback (11)
*** RC-CA: Career Awareness (8)
*** RC-SET: Settings (9)
*** RC-FB: Firebase (12)
*** RC-GAL: Gallery (12)
** Part B: Functionality Tests (107)
*** FN-SP: Splash Screen (3)
*** FN-OB: App Onboarding (6)
*** FN-LG: Login & Auth (6)
*** FN-HM: Home Screen (5)
*** FN-IT: Ishihara Intro (6)
*** FN-TE: Test Execution (17)
*** FN-RS: Results & Scoring (13)
*** FN-CE: Camera Enhancement (14)
*** FN-CI: Color Identifier (6)
*** FN-SIM: CVD Simulation (5)
*** FN-GL: CVD Gallery (4)
*** FN-ED: Education & Articles (6)
*** FN-SV: Survey (3)
*** FN-ST: Settings (8)
*** FN-HI: History (3)
*** FN-AD: Admin & Research (6)
** Part C: Android Core (80)
*** VE: Visual Experience (20)
*** AF: Android Functionality (20)
*** PS: Performance & Stability (20)
*** SC: Privacy & Security (20)
** Part D: Compatibility (37)
*** CM-AV: Android Versions (7)
*** CM-SS: Screen Sizes (8)
*** CM-OR: Orientation (2)
*** CM-CH: Camera Hardware (7)
*** CM-NW: Network Conditions (8)
*** CM-DV: Device Edge Cases (7)
left side
@enduml
```

**Suggested Caption:** Figure 3.X: Test Case Classification Hierarchy

---

## Diagram 3: Screen Coverage Traceability Matrix (Component Diagram)

Maps each application screen to the test case categories that cover it.

```plantuml
@startuml screen_coverage_traceability
!theme plain
skinparam backgroundColor #FFFFFF
skinparam ComponentBackgroundColor #E8EAF6
skinparam ComponentBorderColor #1F3864
skinparam PackageBackgroundColor #F5F5F5
skinparam PackageBorderColor #2E74B5
skinparam ArrowColor #2E74B5
skinparam NoteFontSize 9
skinparam DefaultFontSize 10

title **Screen-to-Test-Case Traceability**

package "Authentication Flow" as auth {
  [SplashScreen] as SP
  [AppOnboarding] as OB
  [LoginScreen] as LG
  [SignUp] as SU
}

package "Main Application" as main_pkg {
  [HomeScreen] as HM
  [ProfileScreen] as PR
  [HistoryScreen] as HI
  [SettingsScreen] as ST
}

package "Ishihara Test Flow" as ish {
  [IshiharaIntroScreen] as II
  [IshiharaOnboarding] as IO
  [TestScreen] as TS
  [ResultsScreen] as RS
  [SurveyScreen] as SV
}

package "Camera Pipeline" as cam {
  [CameraEnhanceScreen] as CE
  [ColorIdentifierScreen] as CI
  [CVDSimulationScreen] as CS
  [CVDGalleryScreen] as GL
}

package "Content Screens" as content {
  [EducationListScreen] as ED
  [ArticleScreen] as AR
  [CareerDetail] as CD
}

package "Admin Portal" as admin {
  [AdminLoginScreen] as AL
  [AdminHubScreen] as AH
  [ResearchDashboard] as RD
}

note right of SP : FN-SP, VE-001, PS-001/002
note right of OB : FN-OB, VE-002
note right of LG : RC-AU, FN-LG, SC-001..003
note right of SU : RC-AU-001..007

note right of HM : RC-NAV, FN-HM
note right of PR #FFF3CD : **Partial** - needs dedicated TCs
note right of HI : FN-HI-001..003
note right of ST : RC-SET, FN-ST, VE-016

note right of II : FN-IT-001..003
note right of IO : FN-IT-004..006
note right of TS : RC-ISH (47), FN-TE (17), VE, PS
note right of RS : FN-RS (13), VE-013

note right of CE : RC-CAM (39), RC-CC (27), FN-CE (14)
note right of CI : RC-CI (20), FN-CI (6), VE-014/015
note right of CS : RC-SIM (7), FN-SIM (5), PS-006/007
note right of GL : RC-GAL (12), FN-GL (4)

note right of CD #FFF3CD : **Partial** - needs detail tests
note right of ED : FN-ED-001..002
note right of AR : FN-ED-003..006, VE-017

note right of AL : FN-AD-001..002
note right of AH : FN-AD-003..004
note right of RD : FN-AD-005/006, VE-019, SC-017

@enduml
```

**Suggested Caption:** Figure 3.X: Screen-to-Test-Case Traceability Matrix

---

## Diagram 4: Testing Execution Workflow (Sequence Diagram)

Shows the step-by-step process a tester follows when executing the test suite.

```plantuml
@startuml testing_execution_workflow
!theme plain
skinparam backgroundColor #FFFFFF
skinparam ParticipantBackgroundColor #E8EAF6
skinparam ParticipantBorderColor #1F3864
skinparam SequenceGroupBackgroundColor #F5F5F5
skinparam SequenceGroupBorderColor #2E74B5
skinparam ArrowColor #1F3864
skinparam NoteBackgroundColor #FFF8E1
skinparam NoteBorderColor #F9A825
skinparam DefaultFontSize 11

title **Test Execution Workflow**

actor Tester as T
participant "Test Device\n(Android)" as D
participant "ReColor\nApplication" as App
participant "Firebase\nBackend" as FB
database "Test Results\nChecklist" as R

== Pre-Execution Setup ==

T -> D : Install APK (debug build)
T -> D : Verify Android version\n& device specs
T -> R : Prepare checklist\n(457 test cases)

== Part A + B: Functional Testing ==

group Module & Functionality Tests [340 TCs]
  T -> App : Execute authentication flows\n(RC-AU, FN-LG)
  App -> FB : Verify Firebase auth
  T -> R : Record Pass/Fail

  T -> App : Navigate all screens\n(RC-NAV, FN-HM)
  T -> R : Record Pass/Fail

  T -> App : Camera pipeline tests\n(RC-CAM, RC-CC, FN-CE)
  note right of App
    Test rear/front camera,
    freeze, capture, correction,
    simulation, identifier
  end note
  T -> R : Record Pass/Fail

  T -> App : Ishihara test execution\n(RC-ISH, FN-TE, FN-RS)
  App -> FB : Verify dual-write
  T -> R : Record Pass/Fail

  T -> App : Remaining features\n(Gallery, Career, Education,\nSurvey, Settings, Admin)
  T -> R : Record Pass/Fail
end

== Part C: Quality Testing ==

group Android Core Tests [80 TCs]
  T -> App : Visual experience audit\n(VE-001..020)
  T -> R : Record Pass/Fail

  T -> App : Android lifecycle tests\n(AF-001..020)
  note right of App
    Background/foreground,
    back button, permissions,
    auth state, input handling
  end note
  T -> R : Record Pass/Fail

  T -> App : Performance benchmarks\n(PS-001..020)
  T -> R : Record latency values\n& Pass/Fail

  T -> App : Privacy & security audit\n(SC-001..020)
  App -> FB : Verify data isolation
  T -> R : Record Pass/Fail
end

== Part D: Compatibility Testing ==

group Compatibility Tests [37 TCs]
  T -> D : Test on Device 1\n(small screen / Android 12)
  T -> R : Record Pass/Fail

  T -> D : Test on Device 2\n(standard / Android 13-14)
  T -> R : Record Pass/Fail

  T -> D : Test on Device 3\n(large / Android 15)
  T -> R : Record Pass/Fail

  T -> D : Network condition tests\n(WiFi, mobile, offline)
  T -> R : Record Pass/Fail
end

== Post-Execution ==

T -> R : Compile summary report
T -> R : Calculate pass rate\n& coverage metrics
T -> R : Document failures\n& recommendations

@enduml
```

**Suggested Caption:** Figure 3.X: Test Execution Workflow

---

## Diagram 5: Quality Dimension Coverage (Deployment Diagram)

Shows the quality dimensions tested and their relationship to the system.

```plantuml
@startuml quality_dimensions
!theme plain
skinparam backgroundColor #FFFFFF
skinparam NodeBackgroundColor #E8EAF6
skinparam NodeBorderColor #1F3864
skinparam ArtifactBackgroundColor #D6E4F0
skinparam ArtifactBorderColor #2E74B5
skinparam CloudBackgroundColor #FFF8E1
skinparam CloudBorderColor #F9A825
skinparam ArrowColor #1F3864
skinparam DefaultFontSize 11

title **Quality Dimension Coverage Model**

node "ReColor Application" as app {
  artifact "24 Screens" as screens
  artifact "7 Components" as comps
  artifact "3 Utilities" as utils
  artifact "tensorHelper.js\n(CVD Algorithms)" as tensor
}

cloud "Firebase Backend" as fb {
  artifact "Authentication" as auth_a
  artifact "Firestore DB" as db
  artifact "Anonymized\nResearch Data" as research
}

rectangle "**Functional Correctness**\n340 test cases (Parts A+B)" as func #E8F5E9 {
  card "Module tests (RC-*): 233" as rc
  card "Screen tests (FN-*): 107" as fn
}

rectangle "**Visual & UX Quality**\n20 test cases (Part C-A)" as visual #E3F2FD {
  card "Animation smoothness" as v1
  card "Color consistency" as v2
  card "Typography hierarchy" as v3
}

rectangle "**Platform Reliability**\n40 test cases (Part C-B,C)" as platform #FFF3E0 {
  card "Android lifecycle (AF): 20" as af
  card "Performance (PS): 20" as ps
}

rectangle "**Security & Privacy**\n20 test cases (Part C-D)" as security #FCE4EC {
  card "Auth security" as s1
  card "Data anonymization" as s2
  card "Permission model" as s3
}

rectangle "**Device Compatibility**\n37 test cases (Part D)" as compat #F3E5F5 {
  card "Android 12-15" as c1
  card "Screen sizes" as c2
  card "Network conditions" as c3
}

func --> app
visual --> app
platform --> app
security --> app : validates
security --> fb : validates
compat --> app

@enduml
```

**Suggested Caption:** Figure 3.X: Quality Dimension Coverage Model

---

## Diagram 6: Ishihara Test Scoring Pipeline (Activity Diagram)

Shows the detailed testing flow for the Ishihara screening module — the most extensively tested feature (64 TCs).

```plantuml
@startuml ishihara_testing_pipeline
!theme plain
skinparam backgroundColor #FFFFFF
skinparam ActivityBackgroundColor #E8EAF6
skinparam ActivityBorderColor #1F3864
skinparam ActivityDiamondBackgroundColor #FFF8E1
skinparam ActivityDiamondBorderColor #F9A825
skinparam ArrowColor #1F3864
skinparam PartitionBackgroundColor #F5F5F5
skinparam PartitionBorderColor #2E74B5
skinparam NoteBackgroundColor #E8F5E9
skinparam NoteBorderColor #2E7D32
skinparam DefaultFontSize 11

title **Ishihara Screening Test Validation Pipeline**
footer Test Cases: RC-ISH-001..047 + FN-IT/TE/RS-001..036

start

partition "Module Launch\n(RC-ISH-001..003, FN-IT)" {
  :Open Ishihara module;
  if (Brightness >= 80%?) then (yes)
    :Show Plate 1;
  else (no)
    :Display brightness\nprompt;
    :User adjusts\nbrightness;
  endif
  :Select test type;
  note right
    Quick: 14 plates
    Comprehensive: 38 plates
    (FN-IT-001..003)
  end note
  :Complete 6-slide\ncalibration onboarding;
}

partition "Plate Display\n(RC-ISH-004..028, FN-TE-001..017)" {
  :Display plate image\n(3-second timer);
  note right
    Plate #1 always first
    Rest randomized
    (FN-TE-001..002)
  end note

  if (Input type?) then (numeric)
    :User enters number\nvia numpad;
    note right
      Max 3 digits
      Backspace supported
      (FN-TE-004..005)
    end note
  else (tracing)
    :User taps\nYes/No button;
    note right
      Plates 26-38
      (FN-TE-007)
    end note
  endif

  :Record answer\n(correct/incorrect);
  :Haptic + speech\nfeedback;
}

partition "Stage 1 Scoring\n(RC-ISH-032..038)" {
  :Evaluate Stage 1\n(plates 1-21);

  if (Score >= 17/21?) then (yes)
    :Diagnosis: **Normal**;
    note right
      Skip Stage 2
      (RC-ISH-032..033)
    end note
  elseif (Score 14-16/21?) then (borderline)
    :Diagnosis:\n**Indeterminate**;
    note right
      Skip Stage 2
      (RC-ISH-034..035)
    end note
  else (CVD indicated)
    :Proceed to\n**Stage 2**;
    note right
      Score <= 13/21
      (RC-ISH-036..038)
    end note
  endif
}

if (Stage 2 needed?) then (yes)
  partition "Stage 2 Classification\n(RC-ISH-039..042)" {
    :Present diagnostic\nplates 22-25;

    if (Protan pattern\n6,2,5,6?) then (yes)
      :Classification:\n**Protanomaly**;
    elseif (Deutan pattern\n2,4,3,9?) then (yes)
      :Classification:\n**Deuteranomaly**;
    else (mixed)
      :Classification:\n**Indeterminate**;
    endif
  }
else (no)
endif

partition "Results & Firebase\n(FN-RS, RC-FB)" {
  :Calculate weighted score\n(diagnostic plates = 2x);
  :Determine severity\n(None/Mild/Moderate/Severe);
  :Save to Firebase;
  note right
    Dual-write:
    1. users/uid/history
    2. research_data_anonymized
    (RC-FB-001..012)
  end note
  :Display results screen\nwith diagnosis;
}

stop

@enduml
```

**Suggested Caption:** Figure 3.X: Ishihara Screening Test Validation Pipeline

---

## Usage Instructions

1. Copy each `@startuml ... @enduml` block
2. Paste into [PlantUML Online Server](https://www.plantuml.com/plantuml/uml) or a local PlantUML renderer
3. Export as PNG (300 DPI recommended for print)
4. Insert into Chapter 3 manuscript with appropriate figure captions

### Suggested Figure Captions

| Diagram | Suggested Caption |
|---------|-------------------|
| 1 | Figure 3.X: Testing Methodology Pipeline |
| 2 | Figure 3.X: Test Case Classification Hierarchy |
| 3 | Figure 3.X: Screen-to-Test-Case Traceability Matrix |
| 4 | Figure 3.X: Test Execution Workflow |
| 5 | Figure 3.X: Quality Dimension Coverage Model |
| 6 | Figure 3.X: Ishihara Screening Test Validation Pipeline |
