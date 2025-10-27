// mongo-init/init.js
// Initialisation unique : crée l'utilisateur applicatif + collection "patients" avec validator
// + index (unique) sur name et index sur admissions.date_of_admission
// NOTE: ce script s'exécute uniquement au 1er démarrage d'un volume de données vierge.

(() => {
  const dbName = process.env.MONGO_INITDB_DATABASE || 'med_db';
  const appUser = process.env.MONGO_APP_USERNAME || 'ingestor';
  const appPwd  = process.env.MONGO_APP_PASSWORD || 'ingestor_password';

  print(`[init.js] Initializing DB '${dbName}' and user '${appUser}'`);
  db = db.getSiblingDB(dbName);

  // 1) Créer l'utilisateur applicatif (moindre privilège possible plus tard)
  try {
    db.createUser({
      user: appUser,
      pwd: appPwd,
      roles: [
        { role: "readWrite", db: dbName },
        { role: "dbOwner", db: dbName }
      ]
    });
    print(`[init.js] User '${appUser}' created on DB '${dbName}'.`);
  } catch (e) {
    print(`[init.js] createUser warning: ${e}`);
  }

  // 2) Créer la collection "patients" avec un validator JSON Schema
  // IMPORTANT : on utilise les noms de champs "normalisés" (snake_case) après rename_map
  //   - name, age, gender, blood_type, medical_condition, insurance_provider
  //   - admissions[].date_of_admission, discharge_date, doctor, hospital, admission_type, room_number, billing_amount, medication, test_results
  try {
    db.createCollection("patients", {
      validator: {
        $jsonSchema: {
          bsonType: "object",
          required: ["name", "admissions"],
          properties: {
            name:                { bsonType: ["string"] },
            age:                 { bsonType: ["int","long","null"], minimum: 0 },
            gender:              { enum: ["M","F","X", null] },
            blood_type:          { bsonType: ["string","null"] },
            medical_condition:   { bsonType: ["string","null"] },
            insurance_provider:  { bsonType: ["string","null"] },
            admissions: {
              bsonType: "array",
              items: {
                bsonType: "object",
                properties: {
                  date_of_admission: { bsonType: ["date","null"] },
                  discharge_date:     { bsonType: ["date","null"] },
                  doctor:             { bsonType: ["string","null"] },
                  hospital:           { bsonType: ["string","null"] },
                  admission_type:     { bsonType: ["string","null"] },
                  room_number:        { bsonType: ["int","long","null"], minimum: 0 },
                  billing_amount:     { bsonType: ["double","int","long","decimal","null"], minimum: 0 },
                  medication:         { bsonType: ["string","null"] },
                  test_results:       { bsonType: ["string","null"] }
                },
                additionalProperties: true
              }
            }
          },
          additionalProperties: true
        }
      }
    });
    print("[init.js] Collection 'patients' created with validator.");
  } catch (e) {
    if (e.codeName === 'NamespaceExists') {
      print("[init.js] Collection 'patients' already exists (skipping).");
    } else {
      throw e;
    }
  }

  // 3) Index
  try {
    db.patients.createIndex({ name: 1 }, { unique: true, sparse: true });
    db.patients.createIndex({ "admissions.date_of_admission": 1 });
    print("[init.js] Indexes created.");
  } catch (e) {
    print(`[init.js] createIndex warning: ${e}`);
  }

  print("[init.js] Initialization complete.");
})();
