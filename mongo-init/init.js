// ------------------------------------------------------------
// init.js  — Création DB + utilisateurs + index (version finale)
// ------------------------------------------------------------

// 1️⃣ Configuration de base
const dbName = "meddb";
const collName = "patients";

// utilisateurs applicatifs
const appUser = "ingest_user";
const appPass = "ingest_pass"; // 🔒 remplace <fort> par un vrai mot de passe robuste
const analystUser = "analyst_read";
const analystPass = "analyst_pass"; // 🔒 idem

// Connexion à la base applicative
const db = db.getSiblingDB(dbName);

// 2️⃣ Création des utilisateurs si absents
try {
  if (!db.getUser(appUser)) {
    print(`[init] Creating user ${appUser} (readWrite)…`);
    db.createUser({
      user: appUser,
      pwd: appPass,
      roles: [{ role: "readWrite", db: dbName }],
    });
  } else {
    print(`[init] User ${appUser} already exists — skipping`);
  }

  if (!db.getUser(analystUser)) {
    print(`[init] Creating user ${analystUser} (read)…`);
    db.createUser({
      user: analystUser,
      pwd: analystPass,
      roles: [{ role: "read", db: dbName }],
    });
  } else {
    print(`[init] User ${analystUser} already exists — skipping`);
  }
} catch (e) {
  print(`[init][WARN] createUser error: ${e}`);
}

// 3️⃣ Création de la collection si absente
try {
  const exists = db.getCollectionNames().includes(collName);
  if (!exists) {
    print(`[init] Creating collection ${collName}…`);
    db.createCollection(collName);
  } else {
    print(`[init] Collection ${collName} already exists — skipping`);
  }
} catch (e) {
  print(`[init][WARN] createCollection error: ${e}`);
}

// 4️⃣ Création des index (unicité + filtres)
try {
  print("[init] Ensuring indexes on patients…");

  // index d’unicité sur le modèle stocké
  db[collName].createIndex(
    { "Name": 1, "Age": 1, "Gender": 1, "Blood Type": 1 },
    { name: "uniq_name_age_gender_blood", unique: true }
  );

  // index utiles
  db[collName].createIndex({ "Name": 1 }, { name: "idx_Name" });
  db[collName].createIndex({ "Blood Type": 1 }, { name: "idx_BloodType" });
  db[collName].createIndex(
    { "Admissions.Date of Admission": -1 },
    { name: "idx_Admissions_DoA_desc" }
  );

  print("[init] Indexes ensured ✅");
} catch (e) {
  print(`[init][ERROR] index creation failed: ${e}`);
}

print("[init] Initialization completed ✅");
