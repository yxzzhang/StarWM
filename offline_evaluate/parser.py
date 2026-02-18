import re
from collections import defaultdict
from data_types import SC2Unit

class SC2TextParser:
    def __init__(self):
        pass

    def parse(self, text):
        """
        Main entry point. Parses the entire observation text into a structured dictionary.
        Returns:
            {
                "info": dict,
                "queue_items": list[dict], # list of unified queue items
                "units": list[SC2Unit],
                "structures": list[SC2Unit],
                "enemies": list[SC2Unit]
            }
        """
        if not text:
            return None

        sections = self._split_sections(text)
        
        info = self._parse_info(sections.get("Info", ""))
        queue_text = sections.get("Queue", "").strip()
        
        # Parse Queue items to extract constructing structures and production orders
        queue_items = self._parse_queue(queue_text)
        
        # Parse My Units (Army only, exclude workers)
        units = self._parse_my_units(sections.get("My Units", ""))
        
        # Parse My Structures
        structures = self._parse_my_structures(sections.get("My Structures", ""))
        
        # Parse Enemies
        enemies = self._parse_visible_hostiles(sections.get("Visible Hostiles", ""))
        
        return {
            "info": info,
            "queue_items": queue_items,
            "units": units,
            "structures": structures,
            "enemies": enemies
        }

    def _split_sections(self, text):
        section_map = {}
        current_header = None
        buffer = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r"^\[(.*?)\]$", line)
            if match:
                if current_header:
                    section_map[current_header] = "\n".join(buffer)
                current_header = match.group(1)
                buffer = []
            else:
                buffer.append(line)
        
        if current_header:
            section_map[current_header] = "\n".join(buffer)
            
        return section_map

    def _parse_info(self, text):
        info = {
            "time_seconds": 0, # Added for plotting
            "minerals": 0, "gas": 0, 
            "minerals_rate": 0, "gas_rate": 0,
            "supply_used": 0, "supply_cap": 0, 
            "army_count": 0, "workers_count": 0,
            "upgrades": set(), "alerts": set(),
            "map_name": "",
            "race": "",
            "enemy_race": ""
        }
        
        for line in text.split('\n'):
            # Time: 08:50 | Race: Terran | Enemy Race: Zerg | Map: Flat64
            if "Time:" in line:
                t_match = re.search(r"Time:\s*(\d+):(\d+)", line)
                if t_match:
                    mins = int(t_match.group(1))
                    secs = int(t_match.group(2))
                    info["time_seconds"] = mins * 60 + secs
                
                # Use robust searching or splitting for other fields
                parts = [p.strip() for p in line.split('|')]
                for p in parts:
                    if p.startswith("Map:"):
                        info["map_name"] = p.split(":")[1].strip()
                    elif p.startswith("Race:"):
                        info["race"] = p.split(":")[1].strip()
                    elif p.startswith("Enemy Race:"):
                        info["enemy_race"] = p.split(":")[1].strip()
                
                # Fallback regex if pipe splitting fails or format is loose
                if not info["map_name"]:
                    map_match = re.search(r"Map:\s*([A-Za-z0-9_]+)", line)
                    if map_match: info["map_name"] = map_match.group(1)

            # Minerals: 140 (+4395/min) | Gas: 85 (+1030/min)
            if "Minerals:" in line:
                parts = line.split('|')
                min_part = parts[0]
                gas_part = parts[1] if len(parts) > 1 else ""
                
                # Minerals
                m_match_verbose = re.search(r"(\d+)\s*\(\+", min_part)
                if m_match_verbose:
                    info["minerals"] = int(m_match_verbose.group(1))
                else:
                    m_match = re.search(r"Minerals:\s*(\d+)", min_part)
                    if m_match: info["minerals"] = int(m_match.group(1))
                
                mr_match = re.search(r"\+(\d+)/min", min_part)
                if mr_match: info["minerals_rate"] = int(mr_match.group(1))
                
                # Gas
                g_match_verbose = re.search(r"(\d+)\s*\(\+", gas_part)
                if g_match_verbose:
                    info["gas"] = int(g_match_verbose.group(1))
                else:
                    g_match = re.search(r"Gas:\s*(\d+)", gas_part)
                    if g_match: info["gas"] = int(g_match.group(1))
                
                gr_match = re.search(r"\+(\d+)/min", gas_part)
                if gr_match: info["gas_rate"] = int(gr_match.group(1))
            
            # Supply: 159/163 (Army: 67, Workers: 88)
            if "Supply:" in line:
                s_match = re.search(r"Supply:\s*(\d+)/(\d+)", line)
                if s_match:
                    info["supply_used"] = int(s_match.group(1))
                    info["supply_cap"] = int(s_match.group(2))
                
                aw_match = re.search(r"\(Army:\s*(\d+),\s*Workers:\s*(\d+)\)", line)
                if aw_match:
                    info["army_count"] = int(aw_match.group(1))
                    info["workers_count"] = int(aw_match.group(2))
            
            # Alerts: None or List
            if "Alerts:" in line:
                try:
                    content = line.split("Alerts:")[1].strip()
                    if content and content != "None":
                        info["alerts"] = {x.strip() for x in content.split(",") if x.strip()}
                except IndexError:
                    pass
            
            # Upgrades: ...
            if "Upgrades:" in line:
                try:
                    content = line.split("Upgrades:")[1].strip()
                    if content and content != "None":
                        info["upgrades"] = {x.strip() for x in content.split(",") if x.strip()}
                except IndexError:
                    pass
                    
        return info

    def _parse_queue(self, text):
        queue_items = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or line == "None": continue
            
            # 1. Construction
            # - Constructing: Factory [608] at (35,41) (39%)
            # Use non-greedy (.*?) for name to allow spaces
            const_match = re.search(r"Constructing:\s*(.*?)\s*\[([\d,]+)\]\s*at\s*\((\d+),\s*(\d+)\)\s*\((\d+(?:\.\d+)?)%\)", line)
            if const_match:
                u_type = const_match.group(1)
                ids_str = const_match.group(2)
                uids = [x.strip() for x in ids_str.split(',') if x.strip()]
                x, y = int(const_match.group(3)), int(const_match.group(4))
                prog = float(const_match.group(5))
                
                for uid in uids:
                    queue_items.append({
                        "type": "construction",
                        "uid": uid,
                        "u_type": u_type,
                        "pos": (x,y),
                        # "progress": prog * 0.01
                        "progress": prog
                    })
                continue
            
            # 2. Production
            # - Orbitalcommand [1] at (24,22): Commandcentertrain_scv (3%)
            prod_match = re.search(r"(.*?)\s*\[([\d,]+)\]\s*at\s*\((\d+),\s*(\d+)\):\s*(.*)", line)
            if prod_match:
                producer_type = prod_match.group(1)
                ids_str = prod_match.group(2)
                producer_ids = [x.strip() for x in ids_str.split(',') if x.strip()]
                px, py = int(prod_match.group(3)), int(prod_match.group(4))
                content = prod_match.group(5)
                
                tasks = re.findall(r"([A-Za-z0-9_]+)\s*\(([^)]+)\)", content)
                
                for pid in producer_ids:
                    for name, val in tasks:
                        prog = 0.0
                        if "Waiting" in val:
                            prog = 0.0
                        elif "%" in val:
                            try:
                                nums = re.findall(r"(\d+(?:\.\d+)?)%", val)
                                if nums:
                                    prog = float(nums[-1])
                                else:
                                    nums = re.findall(r"(\d+(?:\.\d+)?)", val)
                                    if nums:
                                        prog = float(nums[-1])
                            except:
                                print(f"Error parsing progress: {val}")
                        
                        queue_items.append({
                            "type": "production",
                            "producer_type": producer_type,
                            "producer_id": pid,
                            "pos": (px, py),
                            "product": name,
                            # "progress": prog * 0.01
                            "progress": prog
                        })

        return queue_items

    def _parse_my_units(self, text):
        units = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            if "> Workers:" in line:
                continue
            if "> Army:" in line:
                continue
                
            if line.startswith("-"):
                # - Scv [3] at (41,47) (HP:42%, St:Moving)
                match = re.search(r"-\s*(.*?)\s*\[(\d+)\]\s*at\s*\((\d+),\s*(\d+)\)\s*\((.*)\)", line)
                if match:
                    u_type = match.group(1)
                    
                    if "Scv" in u_type or "Mule" in u_type or "Probe" in u_type or "Drone" in u_type:
                        continue
                        
                    uid = match.group(2)
                    x, y = int(match.group(3)), int(match.group(4))
                    details = match.group(5)
                    
                    hp, eng, buff, status = self._parse_details(details)
                    
                    units.append(SC2Unit(
                        uid=uid, u_type=u_type, pos=(x,y),
                        hp=hp, eng=eng, buff=buff, status=status,
                        category="Army"
                    ))

        return units

    def _parse_my_structures(self, text):
        structures = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # - Barracks [44] at (29,33) (HP:100%, St:Training, AddOn:Techlab [101])
            if line.startswith("-"):
                match = re.search(r"-\s*(.*?)\s*\[(\d+)\]\s*at\s*\((\d+),\s*(\d+)\)\s*\((.*)\)", line)
                if match:
                    u_type = match.group(1)
                    uid = match.group(2)
                    x, y = int(match.group(3)), int(match.group(4))
                    details = match.group(5)
                    
                    hp, eng, buff, status = self._parse_details(details)
                    
                    # AddOn Parsing
                    addon = None
                    addon_match = re.search(r"AddOn:(.*?)\s*\[(\d+)\]", details)
                    if addon_match:
                        addon = {"type": addon_match.group(1), "id": int(addon_match.group(2))}
                    
                    cat = "SelfStructure"
                    
                    structures.append(SC2Unit(
                        uid=uid, u_type=u_type, pos=(x,y),
                        hp=hp, eng=eng, buff=buff, status=status,
                        addon=addon, category=cat
                    ))
        return structures

    def _parse_visible_hostiles(self, text):
        enemies = []
        lines = text.split('\n')
        
        current_subcat = "EnemyUnit" # Default
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            if "> Enemy Units:" in line:
                current_subcat = "EnemyUnit"
                continue
            if "> Enemy Structures:" in line:
                current_subcat = "EnemyStruct"
                continue
            if "> Snapshot Enemy Structures:" in line:
                current_subcat = "SnapEnemyStruct"
                continue
            if "> Radar Blips:" in line:
                continue
                
            if line.startswith("-"):
                if current_subcat == "SnapEnemyStruct":
                    # - Extractor at (59,15)
                    match = re.search(r"-\s*(.*?)\s*at\s*\((\d+),\s*(\d+)\)", line)
                    if match:
                        u_type = match.group(1)
                        # Check if ID leaked into Snapshot lines (defensive)
                        # The regex (.*?) might consume "Type [ID]".
                        # But Snapshot usually has no ID.
                        # If ID is present, we might want to capture it or ignore it.
                        # The user's obs2text says id_str = "" for snapshot.
                        
                        x, y = int(match.group(2)), int(match.group(3))
                        
                        enemies.append(SC2Unit(
                            uid=None, u_type=u_type, pos=(x,y),
                            hp=None, eng=None, buff=None, status=None,
                            category=current_subcat
                        ))
                else:
                    # - Overlord [385] at (76,38) (HP:25%, St:Idle)
                    match = re.search(r"-\s*(.*?)\s*\[(\d+)\]\s*at\s*\((\d+),\s*(\d+)\)\s*\((.*)\)", line)
                    if match:
                        u_type = match.group(1)
                        uid = match.group(2)
                        x, y = int(match.group(3)), int(match.group(4))
                        details = match.group(5)
                        
                        hp, eng, buff, status = self._parse_details(details)
                        
                        enemies.append(SC2Unit(
                            uid=uid, u_type=u_type, pos=(x,y),
                            hp=hp, eng=eng, buff=buff, status=status,
                            category=current_subcat
                        ))
        return enemies

    def _parse_details(self, details_str):
        hp = None
        eng = None
        buff = []
        status = None
        
        # HP:42%
        hp_m = re.search(r"HP:(\d+)%", details_str)
        if hp_m: hp = int(hp_m.group(1))
        
        # Eng:55% or Eng:25 (Check both, order doesn't matter much)
        # obs2text (new) is Eng:xx%
        eng_m = re.search(r"Eng:(\d+)", details_str)
        if eng_m: eng = int(eng_m.group(1))
        
        # St:Moving
        st_m = re.search(r"St:([A-Za-z0-9_]+)", details_str)
        if st_m: status = st_m.group(1)
        
        # Buff:A,B
        buff_m = re.search(r"Buff:([A-Za-z0-9_,]+)", details_str)
        if buff_m: 
            buff = [b.strip() for b in buff_m.group(1).split(',') if b.strip()]
            
        return hp, eng, buff, status
