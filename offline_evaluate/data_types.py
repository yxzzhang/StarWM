class SC2Unit:
    def __init__(self, uid, u_type, pos=None, hp=None, eng=None, buff=None, orders=None, progress=None, status=None, addon=None, category=None):
        self.id = int(uid) if uid is not None else None
        self.type = u_type
        self.pos = pos # tuple (x, y)
        self.hp = hp # int
        self.eng = eng # int, Energy
        self.buff = buff if buff is not None else [] # list of str
        self.orders = orders if orders is not None else [] # List of dicts {product: str, progress: float}
        self.progress = progress # float (0.0-100.0)
        self.status = status # str
        self.addon = addon # dict {id: int, type: str}
        self.category = category # Army, SelfStructure, EnemyUnit, EnemyStruct

    def __repr__(self):
        return f"<{self.category} {self.type} {self.id} {self.pos}>"

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "pos": self.pos,
            "hp": self.hp,
            "eng": self.eng,
            "buff": self.buff,
            "orders": self.orders,
            "progress": self.progress,
            "status": self.status,
            "addon": self.addon,
            "category": self.category
        }
