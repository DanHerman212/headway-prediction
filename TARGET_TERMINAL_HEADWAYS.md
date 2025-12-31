# Data Representation: Target Terminal Headways

This document outlines the specific methodology for deriving **Target Terminal Headways** from the MTA GTFS Static feed. This metric represents the "Dispatcher Intent" or the scheduled service frequency at the origin of the line.

## 1. Source Data (GTFS Static)
The calculation relies on the standard GTFS Schedule files (Reference: [gtfs.org](https://gtfs.org/documentation/schedule/reference/)).

| File | Key Fields | Purpose |
| :--- | :--- | :--- |
| `trips.txt` | `trip_id`, `route_id`, `service_id`, `direction_id` | Identifies "A" line trips and their service pattern. |
| `stop_times.txt` | `trip_id`, `stop_id`, `stop_sequence`, `departure_time` | Provides the scheduled time for every stop. |
| `calendar.txt` | `service_id`, `monday`...`sunday`, `start_date`, `end_date` | Defines which `service_id` is active on which day of the week. |
| `calendar_dates.txt` | `service_id`, `date`, `exception_type` | Handles exceptions (e.g., holidays running on Sunday schedule). |

## 2. Algorithm: Calculating Target Headway

### Step 1: Identify Active Trips for a Specific Date
1.  **Input:** A target date (e.g., `2025-12-31`).
2.  **Logic:**
    *   Find all `service_id`s active on this day of the week (from `calendar.txt`).
    *   **Add** `service_id`s included by `calendar_dates.txt` (`exception_type=1`).
    *   **Remove** `service_id`s excluded by `calendar_dates.txt` (`exception_type=2`).
3.  **Filter:** Select all `trip_id`s from `trips.txt` that match:
    *   `route_id == 'A'`
    *   `service_id` is in the active list.
    *   `direction_id` matches the target direction (e.g., `0` for Northbound).

### Step 2: Extract Terminal Departure Times
1.  **Join:** Connect the valid `trip_id`s to `stop_times.txt`.
2.  **Filter:** Select only the **Origin Station** for each trip.
    *   Condition: `stop_sequence == 1` (The first stop).
3.  **Result:** A list of scheduled departure times from the terminal.

### Step 3: Compute Headway
1.  **Sort:** Order the departures chronologically by `departure_time`.
2.  **Calculate Delta:**
    $$Headway_n = DepartureTime_n - DepartureTime_{n-1}$$
3.  **Output:** A time-series of scheduled headways (in minutes) indexed by the departure time.

## 3. Example Output
| Departure Time | Trip ID | Target Headway (min) |
| :--- | :--- | :--- |
| 08:00:00 | A_Trip_1 | - |
| 08:05:00 | A_Trip_2 | 5.0 |
| 08:12:00 | A_Trip_3 | 7.0 |
| 08:17:00 | A_Trip_4 | 5.0 |

## 4. Implementation Notes
*   **Time Format:** GTFS times can exceed 24:00:00 (e.g., 25:30:00 for 1:30 AM the next day). The parser must handle this "service day" logic correctly.
*   **Multiple Terminals:** The A-line has multiple southern terminals (Far Rockaway, Lefferts Blvd, Rockaway Park). We must decide whether to calculate headways for *each* branch separately or only for the shared trunk.
    *   *Recommendation:* Calculate headways at the **shared trunk origin** (e.g., after the branches merge) or treat each branch as a separate input channel.
